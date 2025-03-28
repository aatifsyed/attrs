use core::{
    fmt::{self, Write as _},
    mem,
};
use std::{
    borrow::Cow,
    collections::{BTreeMap, btree_map::Entry},
    rc::Rc,
    sync::Arc,
};

use proc_macro2::{Span, TokenStream};
use syn::{
    Attribute, Ident, Token,
    ext::IdentExt as _,
    parse::{ParseStream, Parser},
};

#[derive(Default, Debug)]
pub struct Attrs<'a> {
    map: BTreeMap<Ident, Attr<'a>>,
}

impl<'a> Attrs<'a> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        Q: ?Sized,
        Ident: PartialEq<Q>,
    {
        self.map.keys().any(|it| it == key)
    }
    #[track_caller]
    pub fn once<K, F>(&mut self, key: K, f: F) -> &mut Self
    where
        K: UnwrapIdent,
        F: 'a + FnOnce(ParseStream<'_>) -> syn::Result<()>,
    {
        self.insert(key, Attr::Once(Once::Some(Box::new(f))))
    }
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
    #[track_caller]
    pub fn many<K, F>(&mut self, key: K, f: F) -> &mut Self
    where
        K: UnwrapIdent,
        F: 'a + FnMut(ParseStream<'_>) -> syn::Result<()>,
    {
        self.insert(key, Attr::Many(Box::new(f)))
    }

    pub fn into_parser(mut self) -> impl FnOnce(ParseStream<'_>) -> syn::Result<()> {
        move |input| self.run(input)
    }

    pub fn as_parser(&mut self) -> impl FnMut(ParseStream<'_>) -> syn::Result<()> {
        |input| self.run(input)
    }

    pub fn run_on<Q>(&mut self, key: &Q, attrs: &[Attribute]) -> syn::Result<()>
    where
        Q: ?Sized,
        Ident: PartialEq<Q>,
    {
        for attr in attrs {
            if attr.path().is_ident(key) {
                attr.parse_args_with(self.as_parser())?
            }
        }
        Ok(())
    }

    pub fn extract_from<Q>(&mut self, key: &Q, attrs: &mut Vec<Attribute>) -> syn::Result<()>
    where
        Q: ?Sized,
        Ident: PartialEq<Q>,
    {
        let mut e = None;
        attrs.retain(|attr| match attr.path().is_ident(key) {
            true => {
                match (e.as_mut(), attr.parse_args_with(self.as_parser())) {
                    (_, Ok(())) => {}
                    (None, Err(e2)) => e = Some(e2),
                    (Some(e1), Err(e2)) => e1.combine(e2),
                }
                false // parsed - discard
            }
            false => true, // not ours - keep it
        });
        e.map(Err).unwrap_or(Ok(()))
    }

    pub fn run(&mut self, input: ParseStream<'_>) -> syn::Result<()> {
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
            if let Err(mut e) = input.parse::<Token![,]>() {
                e.combine(syn::Error::new(e.span(), msg));
                return Err(e);
            }
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

impl Parser for &mut Attrs<'_> {
    type Output = ();
    fn parse2(self, tokens: TokenStream) -> syn::Result<Self::Output> {
        self.as_parser().parse2(tokens)
    }
}

impl Parser for Attrs<'_> {
    type Output = ();
    fn parse2(self, tokens: TokenStream) -> syn::Result<Self::Output> {
        self.into_parser().parse2(tokens)
    }
}

pub trait UnwrapIdent {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident;
}

impl UnwrapIdent for String {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident {
        Ident::new(self, Span::call_site())
    }
}
impl UnwrapIdent for str {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident {
        Ident::new(self, Span::call_site())
    }
}
impl UnwrapIdent for Cow<'_, str> {
    #[track_caller]
    fn unwrap_ident(&self) -> Ident {
        Ident::new(self, Span::call_site())
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
