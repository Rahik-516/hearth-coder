; Rust tag query.
;
; Vendored and reviewed rather than imported (docs/tech-stack.md §6.3).
; Upstream reference: https://github.com/tree-sitter/tree-sitter-rust (MIT)
;
; Capture contract:
;   @definition.<kind>   the whole definition node, for line ranges
;   @name                the identifier naming it
;   @reference.<kind>    a use of a name
;   @import              a `use` declaration
;   @import.module       the path being brought into scope

; ---------------------------------------------------------------- definitions

(function_item
  name: (identifier) @name) @definition.function

(struct_item
  name: (type_identifier) @name) @definition.class

(enum_item
  name: (type_identifier) @name) @definition.enum

(union_item
  name: (type_identifier) @name) @definition.class

(trait_item
  name: (type_identifier) @name) @definition.interface

(type_item
  name: (type_identifier) @name) @definition.type

(mod_item
  name: (identifier) @name) @definition.module

(const_item
  name: (identifier) @name) @definition.const

(static_item
  name: (identifier) @name) @definition.var

(macro_definition
  name: (identifier) @name) @definition.macro

; An `impl` block is named after the type it implements, so its methods are attributed to
; that type rather than to an anonymous block. Without this the nesting parent of every
; method would be missing, and Rust has no other syntax that associates the two.
(impl_item
  type: (type_identifier) @name) @definition.class

; ----------------------------------------------------------------- references

(call_expression
  function: (identifier) @reference.call)

(call_expression
  function: (field_expression
    field: (field_identifier) @reference.call))

; `Type::new(...)` and `module::func(...)` are both scoped identifiers; the final segment
; is the thing being used.
(call_expression
  function: (scoped_identifier
    name: (identifier) @reference.call))

(macro_invocation
  macro: (identifier) @reference.call)

(type_identifier) @reference.type

; -------------------------------------------------------------------- imports

(use_declaration
  argument: (scoped_identifier) @import.module) @import

(use_declaration
  argument: (identifier) @import.module) @import

(use_declaration
  argument: (use_as_clause
    path: (_) @import.module)) @import

(use_declaration
  argument: (scoped_use_list
    path: (_) @import.module)) @import
