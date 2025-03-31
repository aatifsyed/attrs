//! An ergonomic [`Parser`] for `#[attributes]`, built on parser combinators.
//!
//! ```
//! # strum_lite::strum!( #[derive(PartialEq, Debug)] enum Casing { Kebab = "kebab-case", Snake = "snake_case" });
//! # fn main() -> syn::Result<()> {
//! # use syn::*;
//! # use attrs::*;
//! # use quote::ToTokens as _;
//! let mut rename_all = None::<Casing>;
//! let mut untagged = false;
//! let mut deny_unknown_fields = false;
//! let mut path_to_serde: Path = parse_quote!(::serde);
//! let attrs: Vec<Attribute> = parse_quote! {
//!     #[serde(rename_all = "kebab-case", untagged)]
//!     #[serde(crate = "custom::path")]
//! };
//!
//! Attrs::new()
//!     .once("rename_all", with::eq(set::from_str(&mut rename_all)))
//!     .once("untagged", set::flag(&mut untagged))
//!     .once("deny_unknown_fields", set::flag(&mut deny_unknown_fields))
//!     .once("crate", with::eq(on::parse_str(&mut path_to_serde)))
//!     .parse_attrs("serde", &attrs)?;
//!
//! assert_eq!(rename_all, Some(Casing::Kebab));
//! assert!(untagged);
//! assert!(!deny_unknown_fields); // not encountered, so not set
//! assert_eq!(path_to_serde.to_token_stream().to_string(), "custom :: path");
//! # Ok(()) }
//! ```
//!
//! # Guide
//!
//! `#[attributes]` as they are used [in the Rust compiler](https://doc.rust-lang.org/reference/attributes.html#meta-item-attribute-syntax)
//! and [in the wild](https://serde.rs/attributes.html) tend to look like this:
//!
//! ```
//! # const _: &str = stringify! {
//!   #[repr(align(128), C)]
//! //  ^^^^ ^^^^^ ^^^   ^
//! //  path  key (val)  bare key
//!   #[serde(rename_all = "kebab-case", untagged)]
//! // ^^^^^^ ^^^^^^^^^^   ^^^^^^^^^^^^  ^^^^^^^^
//! //  path     key     =      val      bare key
//! # };
//! ```
//!
//! To use this library, create an [`Attrs`],
//! and register different `key`s, each with a parsing function.
//!
//! This library provides many parsing functions, but there are four key kinds:
//! - [`bool`](set::bool) takes a `true` or `false` from the input.
//! - [`from_str`](set::from_str) takes a `".."` string from the input,
//!   before trying to [`FromStr`] it into an object.
//! - [`parse_str`](set::parse_str) takes a `".."` string from the input,
//!   before trying to [`syn::parse`] it into an object.
//! - [`parse`](set::parse) directly tries to [`syn::parse`] the input.
//!
//! Every function takes an `&mut` reference to its destination,
//! which will be filled in when the corresponding `key` is encountered.
//! The [`on`] module acts on direct references,
//! whereas the [`set`] module acts on [`Option`]s, filling them with [`Some`].
//!
//! The main ways to separate a key from its value are provided as combinators [`with`]:
//! - [`with::eq`] take an `=` from the input.
//! - [`with::paren`] take a group `(..)` from the input.
//!
//! You may choose to accept a `key` [`once`](Attrs::once) or [`many`](Attrs::many) times,
//! and you can, of course, write your own parsing functions for whatever syntax you have in mind.

use core::{
    fmt::Display,
    fmt::{self, Write as _},
    mem,
    str::FromStr,
};
use std::{
    borrow::Cow,
    collections::{BTreeMap, btree_map::Entry},
    rc::Rc,
    sync::Arc,
};

use proc_macro2::{Span, TokenStream};
use syn::{
    Attribute, Ident, LitBool, LitStr, Token,
    ext::IdentExt as _,
    parse::{Parse, ParseStream, Parser},
};

/// Ergonomic [`Parser`] for `#[attributes]`.
///
/// See [crate documentation](mod@self) for more.
///
/// ```
/// # fn main() -> syn::Result<()> {
/// # use attrs::*;
/// let mut untagged = false;
/// let mut krate = None::<syn::Path>;
///
/// let parseme: syn::Attribute = syn::parse_quote! {
///     #[serde(untagged, crate = "path::to::serde")]
/// };
///
/// parseme.parse_args_with(
///     Attrs::new()
///         .once("untagged", set::flag(&mut untagged))
///         .once("crate", with::eq(set::parse_str(&mut krate)))
/// )?;
///
/// assert!(krate.is_some() && untagged);
/// # Ok(()) }
/// ```
#[derive(Default, Debug)]
pub struct Attrs<'a> {
    map: BTreeMap<Ident, Attr<'a>>,
}

impl<'a> Attrs<'a> {
    /// Create a new empty parser.
    pub fn new() -> Self {
        Self::default()
    }
    /// Whether `key` already exists in this parser.
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        Q: ?Sized,
        Ident: PartialEq<Q>,
    {
        self.map.keys().any(|it| it == key)
    }
    /// Parse tokens following `key` using `f`, at most once.
    ///
    /// See [crate documentation](mod@self) for more.
    ///
    /// # Panics
    /// - If `key` has already been registered.
    /// - If `key` is an invalid ident.
    #[track_caller]
    pub fn once<K, F>(&mut self, key: K, f: F) -> &mut Self
    where
        K: UnwrapIdent,
        F: 'a + FnOnce(ParseStream<'_>) -> syn::Result<()>,
    {
        self.insert(key, Attr::Once(Once::Some(Box::new(f))))
    }
    /// Parse tokens following `key` using `f`, potentially many times.
    ///
    /// See [crate documentation](mod@self) for more.
    ///
    /// # Panics
    /// - If `key` has already been registered.
    /// - If `key` is an invalid ident.
    #[track_caller]
    pub fn many<K, F>(&mut self, key: K, f: F) -> &mut Self
    where
        K: UnwrapIdent,
        F: 'a + FnMut(ParseStream<'_>) -> syn::Result<()>,
    {
        self.insert(key, Attr::Many(Box::new(f)))
    }
    /// If the key `alias` is encountered, call the parser for `key`.
    ///
    /// See [module documentation](mod@self) for more.
    ///
    /// # Panics
    /// - If `alias` has already been registered.
    /// - If `alias` is an invalid ident.
    /// - If `key` has not been registered.
    #[track_caller]
    pub fn alias<A, K>(&mut self, alias: A, key: K) -> &mut Self
    where
        A: UnwrapIdent,
        K: UnwrapIdent,
    {
        let key = key.unwrap_ident();
        assert!(
            self.contains(&key),
            "`{key}` is not registered (aliases may only be registered after their destination)"
        );
        self.insert(alias, Attr::AliasFor(key.unwrap_ident()))
    }
    /// Parse all the [`Attribute`]s where their path is the given `path`.
    ///
    /// ```
    /// # fn main() -> syn::Result<()> {
    /// # use syn::*;
    /// # use attrs::*;
    ///
    /// let mut rename_all = None::<String>;
    /// let mut untagged = false;
    /// let mut deny_unknown_fields = false;
    /// let attrs: Vec<Attribute> = parse_quote! {
    ///     #[serde(rename_all = "kebab-case", untagged)]
    ///     #[default] // SKIPPED
    ///     #[serde(deny_unknown_fields)]
    /// };
    ///
    /// Attrs::new()
    ///     .once("rename_all", with::eq(set::from_str(&mut rename_all)))
    ///     .once("untagged", set::flag(&mut untagged))
    ///     .once("deny_unknown_fields", set::flag(&mut deny_unknown_fields))
    ///     .parse_attrs("serde", &attrs)?;
    ///
    /// assert!(rename_all.is_some() && untagged && deny_unknown_fields);
    /// # Ok(()) }
    /// ```
    pub fn parse_attrs<Q>(&mut self, path: &Q, attrs: &[Attribute]) -> syn::Result<()>
    where
        Q: ?Sized,
        Ident: PartialEq<Q>,
    {
        for attr in attrs {
            if attr.path().is_ident(path) {
                attr.parse_args_with(&mut *self)?
            }
        }
        Ok(())
    }

    /// Parse and remove all the [`Attribute`]s where their path is the given `path`.
    ///
    /// ```
    /// # fn main() -> syn::Result<()> {
    /// # use syn::*;
    /// # use attrs::*;
    ///
    /// let mut rename_all = None::<String>;
    /// let mut untagged = false;
    /// let mut deny_unknown_fields = false;
    /// let mut attrs: Vec<Attribute> = parse_quote! {
    ///     #[serde(rename_all = "kebab-case", untagged)]
    ///     #[default] // SKIPPED
    ///     #[serde(deny_unknown_fields)]
    /// };
    ///
    /// Attrs::new()
    ///     .once("rename_all", with::eq(set::from_str(&mut rename_all)))
    ///     .once("untagged", set::flag(&mut untagged))
    ///     .once("deny_unknown_fields", set::flag(&mut deny_unknown_fields))
    ///     .extract_from("serde", &mut attrs)?;
    ///
    /// assert!(rename_all.is_some() && untagged && deny_unknown_fields);
    /// assert_eq!(attrs.len(), 1); // `#[default]` is still there
    /// # Ok(()) }
    /// ```
    pub fn extract_from<Q>(&mut self, path: &Q, attrs: &mut Vec<Attribute>) -> syn::Result<()>
    where
        Q: ?Sized,
        Ident: PartialEq<Q>,
    {
        let mut e = None;
        attrs.retain(|attr| match attr.path().is_ident(path) {
            true => {
                match (e.as_mut(), attr.parse_args_with(&mut *self)) {
                    (_, Ok(())) => {}
                    (None, Err(e2)) => e = Some(e2),
                    (Some(e1), Err(e2)) => e1.combine(e2),
                }
                false // parsed - remove from `attrs`
            }
            false => true, // not ours - leave in `attrs`
        });
        e.map(Err).unwrap_or(Ok(()))
    }

    /// Parse the entirety of input as a sequence of registered `key`s,
    /// followed by the appropriate combinator,
    /// separated by commas.
    fn parse(&mut self, input: ParseStream<'_>) -> syn::Result<()> {
        let mut msg = String::new();
        for (ix, key) in self
            .map
            .iter()
            .filter_map(|(k, v)| match v {
                Attr::AliasFor(_) => None,
                Attr::Once(_) | Attr::Many(_) => Some(k),
            })
            .enumerate()
        {
            match ix == 0 {
                true => write!(msg, "Expected one of `{key}`").unwrap(),
                false => write!(msg, ", `{key}`").unwrap(),
            }
        }
        let msg = match msg.is_empty() {
            true => String::from("This parser accepts no arguments"),
            false => msg,
        };
        loop {
            if input.is_empty() {
                break;
            }
            match input.call(Ident::parse_any) {
                Ok(it) => {
                    let mut key = it.unraw();
                    loop {
                        match self.map.get_mut(&key) {
                            Some(attr) => match attr {
                                Attr::AliasFor(redirect) => key = redirect.clone(),
                                Attr::Once(once) => {
                                    match mem::replace(once, Once::Already(it.span())) {
                                        Once::Some(f) => {
                                            f(input)?;
                                            break;
                                        }
                                        Once::Already(already) => {
                                            let mut e =
                                                syn::Error::new(it.span(), "Duplicate argument");
                                            e.combine(syn::Error::new(
                                                already,
                                                "Already used here",
                                            ));
                                            return Err(e);
                                        }
                                    }
                                }
                                Attr::Many(f) => {
                                    f(input)?;
                                    break;
                                }
                            },
                            None => return Err(syn::Error::new(it.span(), msg)),
                        }
                    }
                }
                Err(mut e) => {
                    e.combine(syn::Error::new(e.span(), msg));
                    return Err(e);
                }
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        Ok(())
    }

    #[track_caller]
    fn insert(&mut self, key: impl UnwrapIdent, val: Attr<'a>) -> &mut Self {
        match self.map.entry(key.unwrap_ident()) {
            Entry::Vacant(it) => it.insert(val),
            Entry::Occupied(it) => panic!("duplicate entry for key {}", it.key()),
        };
        self
    }

    fn into_parser(mut self) -> impl FnMut(ParseStream<'_>) -> syn::Result<()> {
        move |input| self.parse(input)
    }
    fn as_parser(&mut self) -> impl FnMut(ParseStream<'_>) -> syn::Result<()> {
        |input| self.parse(input)
    }
}

enum Attr<'a> {
    AliasFor(Ident),
    Once(Once<'a>),
    Many(Box<dyn 'a + FnMut(ParseStream<'_>) -> syn::Result<()>>),
}

impl fmt::Debug for Attr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Attr").finish_non_exhaustive()
    }
}

enum Once<'a> {
    Some(Box<dyn 'a + FnOnce(ParseStream<'_>) -> syn::Result<()>>),
    Already(Span),
}

/// Borrow from this object as a [`Parser`].
///
/// ```
/// # fn main() -> syn::Result<()> {
/// # use attrs::*;
/// # use syn::*;
/// use syn::parse::Parser as _;
///
/// let mut untagged = false;
/// let mut krate = None::<Path>;
///
/// Attrs::new()
///     .once("untagged", set::flag(&mut untagged))
///     .once("crate", with::eq(set::parse_str(&mut krate)))
///     .parse_str(r#"untagged, crate = "path::to::serde""#)?;
///
/// assert!(krate.is_some() && untagged);
/// # Ok(()) }
/// ```
impl Parser for &mut Attrs<'_> {
    type Output = ();
    fn parse2(self, tokens: TokenStream) -> syn::Result<Self::Output> {
        self.as_parser().parse2(tokens)
    }
}

/// Move this object into a [`Parser`].
///
/// ```
/// # fn main() -> syn::Result<()> {
/// # use attrs::*;
/// # use syn::*;
/// use syn::parse::Parser as _;
///
/// let mut untagged = false;
/// let mut krate = None::<Path>;
///
/// let mut attrs = Attrs::new();
/// attrs
///     .once("untagged", set::flag(&mut untagged))
///     .once("crate", with::eq(set::parse_str(&mut krate)));
/// attrs.parse_str(r#"untagged, crate = "path::to::serde""#)?;
///
/// assert!(krate.is_some() && untagged);
/// # Ok(()) }
/// ```
impl Parser for Attrs<'_> {
    type Output = ();
    fn parse2(self, tokens: TokenStream) -> syn::Result<Self::Output> {
        self.into_parser().parse2(tokens)
    }
}

#[test]
fn test() {
    use quote::quote;
    use syn::{punctuated::Punctuated, *};

    strum_lite::strum! {
        #[derive(PartialEq, Debug)]
        enum Casing {
            Pascal = "PascalCase",
            Snake = "snake_case",
        }
    }

    let mut casing = Casing::Snake;
    let mut vis = Visibility::Inherited;
    let mut opt_pred = None::<WherePredicate>;
    let mut use_unsafe = false;
    let mut aliases = vec![];

    Attrs::new()
        // `rename_all = "snake_case"`
        .once("rename_all", with::eq(on::from_str(&mut casing)))
        // `vis = pub` or `vis(pub)`
        .once("vis", with::peq(on::parse(&mut vis)))
        // `use_unsafe`
        .once("use_unsafe", set::flag(&mut use_unsafe))
        // `where(T: Foo)`
        .once("where", with::paren(set::parse(&mut opt_pred)))
        // `alias("hello", "world"), alias("goodbye")`
        .many(
            "alias",
            with::paren(|input| {
                aliases.extend(Punctuated::<LitStr, Token![,]>::parse_separated_nonempty(
                    input,
                )?);
                Ok(())
            }),
        )
        .parse2(quote! {
            rename_all = "PascalCase",
            vis = pub,
            use_unsafe,
            where(T: Ord),
            alias("hello", "world"),
            alias("goodbye")
        })
        .unwrap();
    assert_eq!(casing, Casing::Pascal);
    assert!(matches!(vis, Visibility::Public(_)));
    assert!(opt_pred.is_some());
    assert!(use_unsafe);
    assert_eq!(aliases.len(), 3);
}

/// Conversion to an [`Ident`].
///
/// This is primarily an ergonomic aid, and SHOULD NOT be used on untrusted inputs.
pub trait UnwrapIdent {
    /// # Panics
    /// - Implementors may decide to panic.
    #[track_caller]
    fn unwrap_ident(&self) -> Ident;
}

impl UnwrapIdent for str {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident {
        Ident::new(self, Span::call_site())
    }
}
impl UnwrapIdent for String {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident {
        <str>::unwrap_ident(self)
    }
}
impl UnwrapIdent for Cow<'_, str> {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident {
        <str>::unwrap_ident(self)
    }
}
impl UnwrapIdent for Ident {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident {
        self.clone()
    }
}
impl<T: UnwrapIdent + ?Sized> UnwrapIdent for &T {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident {
        T::unwrap_ident(self)
    }
}
impl<T: UnwrapIdent + ?Sized> UnwrapIdent for Box<T> {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident {
        T::unwrap_ident(self)
    }
}
impl<T: UnwrapIdent + ?Sized> UnwrapIdent for Rc<T> {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident {
        T::unwrap_ident(self)
    }
}
impl<T: UnwrapIdent + ?Sized> UnwrapIdent for Arc<T> {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident {
        T::unwrap_ident(self)
    }
}

/// Wrap parsing functions so that they are e.g preceded by `=` or surrounded by `(..)`.
pub mod with {
    use syn::{
        Token, braced, bracketed, parenthesized,
        parse::{ParseStream, discouraged::AnyDelimiter},
        token,
    };

    /// Take an `=` before appling `f`.
    ///
    /// Users should be careful that `f` doesn't consume too far into the input.
    pub fn eq<'a, F>(mut f: F) -> impl 'a + FnMut(ParseStream<'_>) -> syn::Result<()>
    where
        F: 'a + FnMut(ParseStream<'_>) -> syn::Result<()>,
    {
        move |input| {
            input.parse::<Token![=]>()?;
            f(input)
        }
    }
    /// Take a `(...)`, appling `f` to its contents.
    pub fn paren<'a, F>(mut f: F) -> impl 'a + FnMut(ParseStream<'_>) -> syn::Result<()>
    where
        F: 'a + FnMut(ParseStream<'_>) -> syn::Result<()>,
    {
        move |input| {
            let content;
            parenthesized!(content in input);
            f(&content)
        }
    }
    /// Take a `[...]`, appling `f` to its contents.
    pub fn bracket<'a, F>(mut f: F) -> impl 'a + FnMut(ParseStream<'_>) -> syn::Result<()>
    where
        F: 'a + FnMut(ParseStream<'_>) -> syn::Result<()>,
    {
        move |input| {
            let content;
            bracketed!(content in input);
            f(&content)
        }
    }
    /// Take a `{...}`, appling `f` to its contents.
    pub fn brace<'a, F>(mut f: F) -> impl 'a + FnMut(ParseStream<'_>) -> syn::Result<()>
    where
        F: 'a + FnMut(ParseStream<'_>) -> syn::Result<()>,
    {
        move |input| {
            let content;
            braced!(content in input);
            f(&content)
        }
    }
    /// Take any group (`(...)`, `[...]`, `{...}`), appling `f` to its contents.
    pub fn delim<'a, F>(mut f: F) -> impl 'a + FnMut(ParseStream<'_>) -> syn::Result<()>
    where
        F: 'a + FnMut(ParseStream<'_>) -> syn::Result<()>,
    {
        move |input| {
            let (_, _, content) = input.parse_any_delimiter()?;
            f(&content)
        }
    }

    /// Either:
    /// - Take an `=` before applying `f`.
    ///   See also [`eq`].
    /// - Take a `(...)` before applying `f` to its contents.
    pub fn peq<'a, F>(mut f: F) -> impl 'a + FnMut(ParseStream<'_>) -> syn::Result<()>
    where
        F: 'a + FnMut(ParseStream<'_>) -> syn::Result<()>,
    {
        move |input| {
            if input.peek(Token![=]) {
                input.parse::<Token![=]>()?;
                f(input)
            } else if input.peek(token::Paren) {
                let content;
                parenthesized!(content in input);
                f(&content)
            } else {
                Err(input.error("Expected a `=` or `(..)`"))
            }
        }
    }
}

/// Create [`Parser`]s that write to [`&mut Option<T>`](Option) or set [`bool`]s when they parse.
pub mod set {
    use super::*;

    /// Ignores the input, and just sets `dst` to `true`.
    pub fn flag(dst: &mut bool) -> impl '_ + FnMut(ParseStream<'_>) -> syn::Result<()> {
        |_| {
            *dst = true;
            Ok(())
        }
    }
    /// Parse a [`LitBool`] from `input`, assigning it to `dst` in [`Some`].
    pub fn bool(dst: &mut Option<bool>) -> impl '_ + FnMut(ParseStream<'_>) -> syn::Result<()> {
        |input| parse::set::bool(dst, input)
    }
    /// Parse a [`Parse`]-able from `input`, assigning it to `dst` in [`Some`].
    pub fn parse<T: Parse>(
        dst: &mut Option<T>,
    ) -> impl '_ + FnMut(ParseStream<'_>) -> syn::Result<()> {
        |input| parse::set::parse(dst, input)
    }
    /// 1. Parse a [`LitStr`] from `input`.
    /// 2. Parse the contents of that string using [`FromStr`].
    /// 3. Assign the result to `dst` in [`Some`].
    pub fn from_str<T: FromStr>(
        dst: &mut Option<T>,
    ) -> impl '_ + FnMut(ParseStream<'_>) -> syn::Result<()>
    where
        T::Err: Display,
    {
        |input| parse::set::from_str(dst, input)
    }
    /// 1. Parse a [`LitStr`] from `input`.
    /// 2. Parse the contents of that string into the [`Parse`]-able.
    /// 3. Assign the result to `dst` in [`Some`].
    pub fn parse_str<T: Parse>(
        dst: &mut Option<T>,
    ) -> impl '_ + FnMut(ParseStream<'_>) -> syn::Result<()> {
        |input| parse::set::parse_str(dst, input)
    }
}

/// Create [`Parser`]s that write to `&mut T` when they parse.
pub mod on {
    use super::*;

    /// Parse a [`LitBool`], assigning its value to `dst`.
    pub fn bool(dst: &mut bool) -> impl '_ + FnMut(ParseStream<'_>) -> syn::Result<()> {
        |input| parse::bool(dst, input)
    }
    /// Parse a [`Parse`]-able, assigning its value to `dst`.
    pub fn parse<T: Parse>(dst: &mut T) -> impl '_ + FnMut(ParseStream<'_>) -> syn::Result<()> {
        |input| parse::parse(dst, input)
    }
    /// 1. Parse a [`LitStr`].
    /// 2. Parse the contents of that string using [`FromStr`].
    /// 3. Assign the result to `dst`.
    pub fn from_str<T: FromStr>(dst: &mut T) -> impl '_ + FnMut(ParseStream<'_>) -> syn::Result<()>
    where
        T::Err: Display,
    {
        |input| parse::from_str(dst, input)
    }
    /// 1. Parse a [`LitStr`].
    /// 2. Parse the contents of that string into the [`Parse`]-able.
    /// 3. Assign the result to `dst`.
    pub fn parse_str<T: Parse>(dst: &mut T) -> impl '_ + FnMut(ParseStream<'_>) -> syn::Result<()> {
        |input| parse::parse_str(dst, input)
    }
}

/// Straightforward parsing functions.
///
/// Useful for constructing your own leaf combinators.
pub mod parse {
    use super::*;

    /// Parse a [`LitBool`] from `input`, assigning it to `dst`.
    pub fn bool(dst: &mut bool, input: ParseStream<'_>) -> syn::Result<()> {
        *dst = input.parse::<LitBool>()?.value;
        Ok(())
    }
    /// Parse a [`Parse`]-able from `input`, assigning it to `dst`.
    pub fn parse<T: Parse>(dst: &mut T, input: ParseStream<'_>) -> syn::Result<()> {
        *dst = input.parse()?;
        Ok(())
    }
    /// 1. Parse a [`LitStr`] from `input`.
    /// 2. Parse the contents of that string using [`FromStr`].
    /// 3. Assign the result to `dst`.
    pub fn from_str<T: FromStr>(dst: &mut T, input: ParseStream<'_>) -> syn::Result<()>
    where
        T::Err: Display,
    {
        let lit_str = input.parse::<LitStr>()?;
        match lit_str.value().parse() {
            Ok(it) => {
                *dst = it;
                Ok(())
            }
            Err(e) => Err(syn::Error::new(lit_str.span(), e)),
        }
    }
    /// 1. Parse a [`LitStr`] from `input`.
    /// 2. Parse the contents of that string into the [`Parse`]-able.
    /// 3. Assign the result to `dst`.
    pub fn parse_str<T: Parse>(dst: &mut T, input: ParseStream<'_>) -> syn::Result<()> {
        let lit_str = input.parse::<LitStr>()?;
        *dst = T::parse.parse_str(&lit_str.value())?;
        Ok(())
    }

    /// Straightforward parsing functions that set [`Option`]s.
    pub mod set {
        use super::*;

        /// Parse a [`LitBool`] from `input`, assigning it to `dst` in [`Some`].
        pub fn bool(dst: &mut Option<bool>, input: ParseStream<'_>) -> syn::Result<()> {
            *dst = Some(input.parse::<LitBool>()?.value);
            Ok(())
        }
        /// Parse a [`Parse`]-able from `input`, assigning it to `dst` in [`Some`].
        pub fn parse<T: Parse>(dst: &mut Option<T>, input: ParseStream<'_>) -> syn::Result<()> {
            *dst = Some(input.parse()?);
            Ok(())
        }
        /// 1. Parse a [`LitStr`] from `input`.
        /// 2. Parse the contents of that string using [`FromStr`].
        /// 3. Assign the result to `dst` in [`Some`].
        pub fn from_str<T: FromStr>(dst: &mut Option<T>, input: ParseStream<'_>) -> syn::Result<()>
        where
            T::Err: Display,
        {
            let lit_str = input.parse::<LitStr>()?;
            match lit_str.value().parse() {
                Ok(it) => {
                    *dst = Some(it);
                    Ok(())
                }
                Err(e) => Err(syn::Error::new(lit_str.span(), e)),
            }
        }
        /// 1. Parse a [`LitStr`] from `input`.
        /// 2. Parse the contents of that string into the [`Parse`]-able.
        /// 3. Assign the result to `dst` in [`Some`].
        pub fn parse_str<T: Parse>(dst: &mut Option<T>, input: ParseStream<'_>) -> syn::Result<()> {
            *dst = Some(input.parse::<LitStr>()?.parse()?);
            Ok(())
        }
    }
}
