; Java tag query.
;
; Vendored and reviewed rather than imported (docs/tech-stack.md §6.3).
; Upstream reference: https://github.com/tree-sitter/tree-sitter-java (MIT)
;
; Capture contract:
;   @definition.<kind>   the whole definition node, for line ranges
;   @name                the identifier naming it
;   @reference.<kind>    a use of a name
;   @import              an import statement
;   @import.module       the type or package being imported

; ---------------------------------------------------------------- definitions

(class_declaration
  name: (identifier) @name) @definition.class

(interface_declaration
  name: (identifier) @name) @definition.interface

(enum_declaration
  name: (identifier) @name) @definition.enum

(record_declaration
  name: (identifier) @name) @definition.class

(annotation_type_declaration
  name: (identifier) @name) @definition.interface

(method_declaration
  name: (identifier) @name) @definition.method

(constructor_declaration
  name: (identifier) @name) @definition.method

; Fields only. A local variable is declared by `local_variable_declaration`, a different
; node, so locals stay out of the symbol table without needing a scope check.
(field_declaration
  declarator: (variable_declarator
    name: (identifier) @name)) @definition.var

; ----------------------------------------------------------------- references

(method_invocation
  name: (identifier) @reference.call)

(object_creation_expression
  type: (type_identifier) @reference.call)

(type_identifier) @reference.type

; -------------------------------------------------------------------- imports

(import_declaration
  (scoped_identifier) @import.module) @import

(import_declaration
  (identifier) @import.module) @import
