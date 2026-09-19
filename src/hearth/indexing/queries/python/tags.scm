; Python tag query.
;
; Vendored and reviewed rather than imported: this query is effectively the schema of
; Hearth's symbol graph, so an upstream change must never alter symbol extraction
; silently (docs/tech-stack.md §6.3).
;
; Source: adapted from the tree-sitter-python tags query, extended with the captures
; Hearth needs (imports with their module spec, and attribute references).
; Upstream: https://github.com/tree-sitter/tree-sitter-python (MIT)
;
; Capture contract:
;   @definition.<kind>   the whole definition node, for line ranges
;   @name                the identifier naming it
;   @reference.<kind>    a use of a name
;   @import              an import statement
;   @import.module       the module being imported from
;   @import.name         an imported binding

; ---------------------------------------------------------------- definitions

(class_definition
  name: (identifier) @name) @definition.class

(function_definition
  name: (identifier) @name) @definition.function

; Module-level constants: NAME = value at the top level only, so local variables inside
; functions do not flood the symbol table. The capture sits on the assignment, not the
; module, or every constant would report the whole file as its span.
(module
  (expression_statement
    (assignment
      left: (identifier) @name) @definition.constant))

; Annotated module-level bindings: X: SomeType = ...
(module
  (expression_statement
    (assignment
      left: (identifier) @name
      type: (type)) @definition.type))

; ----------------------------------------------------------------- references

(call
  function: (identifier) @name) @reference.call

(call
  function: (attribute
    attribute: (identifier) @name)) @reference.call

; Base classes and type annotations are type references.
(class_definition
  superclasses: (argument_list
    (identifier) @name)) @reference.type

(typed_parameter
  type: (type (identifier) @name)) @reference.type

(type (identifier) @name) @reference.type

; Attribute access that is not a call: obj.field
(attribute
  attribute: (identifier) @name) @reference.attribute

; -------------------------------------------------------------------- imports

; Module and imported name are captured in the SAME pattern, so every match is
; self-contained. Splitting them across patterns loses the pairing: query matches are not
; guaranteed to interleave in source order, so names end up attributed to the wrong module.

; import x   /   import x as y
(import_statement
  name: (dotted_name) @import.module) @import

(import_statement
  name: (aliased_import
    name: (dotted_name) @import.module)) @import

; from x import a
(import_from_statement
  module_name: (_) @import.module
  name: (dotted_name) @import.name) @import

; from x import a as b
(import_from_statement
  module_name: (_) @import.module
  name: (aliased_import
    name: (dotted_name) @import.name)) @import

; from x import *
(import_from_statement
  module_name: (_) @import.module
  (wildcard_import) @import.name) @import
