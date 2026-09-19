; TypeScript tag query.
;
; Vendored and reviewed (docs/tech-stack.md §6.3). The TypeScript grammar is a superset of
; JavaScript, so this repeats the JS patterns and adds the type-level declarations that
; carry most of the meaning in a TS codebase: interfaces, type aliases and enums.
; Upstream: https://github.com/tree-sitter/tree-sitter-typescript (MIT)
;
; The identical file is vendored for `tsx`. Keeping two real files rather than a symlink
; means a reviewer sees exactly what each language runs.

; ---------------------------------------------------------------- definitions

(class_declaration
  name: (type_identifier) @name) @definition.class

(abstract_class_declaration
  name: (type_identifier) @name) @definition.class

(interface_declaration
  name: (type_identifier) @name) @definition.interface

(type_alias_declaration
  name: (type_identifier) @name) @definition.type

(enum_declaration
  name: (identifier) @name) @definition.type

(function_declaration
  name: (identifier) @name) @definition.function

(generator_function_declaration
  name: (identifier) @name) @definition.function

(function_signature
  name: (identifier) @name) @definition.function

(method_definition
  name: (property_identifier) @name) @definition.method

(method_signature
  name: (property_identifier) @name) @definition.method

(public_field_definition
  name: (property_identifier) @name
  value: [(arrow_function) (function_expression)]) @definition.method

(variable_declarator
  name: (identifier) @name
  value: [(arrow_function) (function_expression)]) @definition.function

(lexical_declaration
  (variable_declarator
    name: (identifier) @name
    value: [(number) (string) (template_string) (object) (array) (true) (false)])) @definition.constant

; ----------------------------------------------------------------- references

(call_expression
  function: (identifier) @name) @reference.call

(call_expression
  function: (member_expression
    property: (property_identifier) @name)) @reference.call

(new_expression
  constructor: (identifier) @name) @reference.type

(type_annotation
  (type_identifier) @name) @reference.type

(generic_type
  name: (type_identifier) @name) @reference.type

(extends_clause
  value: (identifier) @name) @reference.type

(implements_clause
  (type_identifier) @name) @reference.type

(member_expression
  property: (property_identifier) @name) @reference.attribute

; -------------------------------------------------------------------- imports

(import_statement
  (import_clause (identifier) @import.name)
  source: (string) @import.module) @import

(import_statement
  (import_clause
    (named_imports
      (import_specifier name: (identifier) @import.name)))
  source: (string) @import.module) @import

(import_statement
  (import_clause
    (namespace_import (identifier) @import.name))
  source: (string) @import.module) @import

(import_statement
  source: (string) @import.module) @import
