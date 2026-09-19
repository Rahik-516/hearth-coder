; JavaScript tag query.
;
; Vendored and reviewed (docs/tech-stack.md §6.3). Adapted from the tree-sitter-javascript
; tags query, extended with imports and the export marker Hearth uses for `exported`.
; Upstream: https://github.com/tree-sitter/tree-sitter-javascript (MIT)
;
; As in the Python query, each import pattern captures its module and name TOGETHER, so
; the pairing cannot drift between matches.

; ---------------------------------------------------------------- definitions

(class_declaration
  name: (identifier) @name) @definition.class

(function_declaration
  name: (identifier) @name) @definition.function

(generator_function_declaration
  name: (identifier) @name) @definition.function

(method_definition
  name: (property_identifier) @name) @definition.method

; const foo = () => {}   /   const foo = function () {}
(variable_declarator
  name: (identifier) @name
  value: [(arrow_function) (function_expression)]) @definition.function

; const X = <literal>  — module-level constants
(lexical_declaration
  (variable_declarator
    name: (identifier) @name
    value: [(number) (string) (template_string) (object) (array) (true) (false)])) @definition.constant

; class fields holding functions
(field_definition
  property: (property_identifier) @name
  value: [(arrow_function) (function_expression)]) @definition.method

; ----------------------------------------------------------------- references

(call_expression
  function: (identifier) @name) @reference.call

(call_expression
  function: (member_expression
    property: (property_identifier) @name)) @reference.call

(new_expression
  constructor: (identifier) @name) @reference.type

(member_expression
  property: (property_identifier) @name) @reference.attribute

; -------------------------------------------------------------------- imports

; import x from "mod"
(import_statement
  (import_clause (identifier) @import.name)
  source: (string) @import.module) @import

; import { a, b } from "mod"
(import_statement
  (import_clause
    (named_imports
      (import_specifier name: (identifier) @import.name)))
  source: (string) @import.module) @import

; import * as ns from "mod"
(import_statement
  (import_clause
    (namespace_import (identifier) @import.name))
  source: (string) @import.module) @import

; import "mod"  — side-effect only
(import_statement
  source: (string) @import.module) @import

; require("mod")
(call_expression
  function: (identifier) @_require
  arguments: (arguments (string) @import.module))
  (#eq? @_require "require")
