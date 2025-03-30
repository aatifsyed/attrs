<!-- cargo-rdme start -->

An ergonomic [`Parser`] library for `#[attributes]`, built on parser combinators.

```rust
let mut rename_all = None::<Casing>;
let mut untagged = false;
let mut deny_unknown_fields = false;
let mut path_to_serde: Path = parse_quote!(::serde);
let attrs: Vec<Attribute> = parse_quote! {
    #[serde(rename_all = "kebab-case", untagged)]
    #[serde(crate = "custom::path")]
};

Attrs::new()
    .once("rename_all", with::eq(set::from_str(&mut rename_all)))
    .once("untagged", set::flag(&mut untagged))
    .once("deny_unknown_fields", set::flag(&mut deny_unknown_fields))
    .once("crate", with::eq(on::parse_str(&mut path_to_serde)))
    .parse_attrs("serde", &attrs)?;

assert_eq!(rename_all, Some(Casing::Kebab));
assert!(untagged);
assert!(!deny_unknown_fields); // not encountered, so not set
assert_eq!(path_to_serde.to_token_stream().to_string(), "custom :: path");
```

# Guide

`#[attributes]` as they are used [in the Rust compiler](https://doc.rust-lang.org/reference/attributes.html#meta-item-attribute-syntax)
and [in the wild](https://serde.rs/attributes.html) tend to look like this:


You register different `key`s with an [`Attrs`] parser, along with a parsing function.

This library provides several parsing functions, but there are four key kinds:
- [`bool`](set::bool) takes a `true` or `false` from the input.
- [`from_str`](set::from_str) takes a `".."` string from the input,
  before trying to [`FromStr`] it into an object.
- [`parse_str`](set::parse_str) takes a `".."` string from the input,
  before trying to [`syn::parse`] it into an object.
- [`parse`](set::parse) directly tries to [`syn::parse`] the input.

Every function takes an `&mut` reference to its destination,
which will be filled in when the corresponding `key` is encountered.
The [`on`] module acts on direct references,
whereas the [`set`] module acts on [`Option`]s, filling them with [`Some`].

The main ways to separate a key from its value are provided as combinators [`with`]:
- [`with::eq`] take an `=` from the input.
- [`with::paren`] take a group `(..)` from the input.

You may choose to accept a `key` [`once`](Attrs::once) or [`many`](Attrs::many) times,
and you can, of course, write your own parsing functions for whatever syntax you have in mind.

<!-- cargo-rdme end -->
