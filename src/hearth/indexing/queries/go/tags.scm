; Go tag query.
;
; Vendored and reviewed rather than imported, for the reason the Python query gives: this
; file is effectively the schema of Hearth's symbol graph for Go, and an upstream change
; must never alter symbol extraction silently (docs/tech-stack.md §6.3).
;
; Upstream reference: https://github.com/tree-sitter/tree-sitter-go (MIT)
;
; Capture contract (identical across languages):
;   @definition.<kind>   the whole definition node, for line ranges
;   @name                the identifier naming it
;   @reference.<kind>    a use of a name
;   @import              an import statement
;   @import.module       the module being imported from

; ---------------------------------------------------------------- definitions

(function_declaration
  name: (identifier) @name) @definition.function

; A method's receiver is what makes it a method rather than a function, but the receiver
; type is not captured as the name: `func (s *Server) Start()` is `Start`, and attaching
; the type here would make every method of every type collide on lookup.
(method_declaration
  name: (field_identifier) @name) @definition.method

(type_declaration
  (type_spec
    name: (type_identifier) @name
    type: (struct_type))) @definition.class

(type_declaration
  (type_spec
    name: (type_identifier) @name
    type: (interface_type))) @definition.interface

; Any other named type: aliases, function types, defined types over primitives.
(type_declaration
  (type_spec
    name: (type_identifier) @name)) @definition.type

(const_declaration
  (const_spec
    name: (identifier) @name)) @definition.const

; Package-level vars only. Locals are `short_var_declaration`, a different node, so this
; does not need a scope guard the way Python's constant rule does.
(var_declaration
  (var_spec
    name: (identifier) @name)) @definition.var

; ----------------------------------------------------------------- references

(call_expression
  function: (identifier) @reference.call)

; `pkg.Fn()` and `obj.Method()` are the same node shape; the field is what is being used,
; and recording it unqualified is what lets the graph link a call to a definition
; elsewhere.
(call_expression
  function: (selector_expression
    field: (field_identifier) @reference.call))

(type_identifier) @reference.type

; -------------------------------------------------------------------- imports

(import_declaration
  (import_spec
    path: (interpreted_string_literal) @import.module)) @import

(import_declaration
  (import_spec_list
    (import_spec
      path: (interpreted_string_literal) @import.module))) @import
