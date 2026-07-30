

<!-- cargo-rdme start -->

Un [`Parser`] ergonómico para `#[attributes]`, construido sobre combinadores de parsers.

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

# Guía

Los `#[attributes]` tal como se utilizan [en el compilador de Rust](https://doc.rust-lang.org/reference/attributes.html#meta-item-attribute-syntax)
y [en la práctica](https://serde.rs/attributes.html) suelen tener el siguiente aspecto:

```rust
  #[repr(align(128), C)]
//  ^^^^ ^^^^^ ^^^   ^
//  path  key (val)  bare key

  #[serde(rename_all = "kebab-case", untagged)]
// ^^^^^^ ^^^^^^^^^^   ^^^^^^^^^^^^  ^^^^^^^^
//  path     key     =      val      bare key
```

Para utilizar esta biblioteca, crea una instancia de [`Attrs`]
y registra diferentes `key`, cada una con una función de análisis.

Esta biblioteca proporciona muchas funciones de análisis, pero existen cuatro tipos principales:
- [`lit`](set::lit) toma un literal como `true` o `100` desde la entrada.
- [`from_str`](set::from_str) toma una cadena `".."` desde la entrada,
  antes de intentar convertirla en un objeto mediante [`FromStr`].
- [`parse_str`](set::parse_str) toma una cadena `".."` desde la entrada,
  antes de intentar analizarla en un objeto mediante [`syn::parse`].
- [`parse`](set::parse) intenta directamente analizar la entrada mediante [`syn::parse`].

Cada función recibe una referencia `&mut` a su destino,
la cual se llenará cuando se encuentre la `key` correspondiente.
El módulo [`on`] opera sobre referencias directas,
mientras que el módulo [`set`] opera sobre [`Option`]s, llenándolas con [`Some`].

Las principales formas de separar una `key` de su valor se proporcionan como combinadores en el módulo [`with`]:
- [`with::eq`] consume un `=` desde la entrada.
- [`with::paren`] consume un grupo `(..)` desde la entrada.

Puedes elegir aceptar una `key` [`once`](Attrs::once) o [`many`](Attrs::many) veces,
y por supuesto, puedes escribir tus propias funciones de análisis para cualquier sintaxis que tengas en mente.

<!-- cargo-rdme end -->
