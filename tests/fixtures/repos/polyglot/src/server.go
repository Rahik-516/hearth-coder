package server

import (
	"errors"
	"fmt"
)

// ErrClosed is returned once the server has shut down.
var ErrClosed = errors.New("server closed")

const DefaultPort = 8080

// Handler responds to a request.
type Handler interface {
	Serve(path string) (string, error)
}

// Server routes requests to handlers.
type Server struct {
	port     int
	handlers map[string]Handler
}

// NewServer builds a server bound to a port.
func NewServer(port int) *Server {
	return &Server{port: port, handlers: make(map[string]Handler)}
}

// Register attaches a handler to a path.
func (s *Server) Register(path string, handler Handler) {
	s.handlers[path] = handler
}

// Serve dispatches one request.
func (s *Server) Serve(path string) (string, error) {
	handler, ok := s.handlers[path]
	if !ok {
		return "", fmt.Errorf("no handler for %s", path)
	}
	return handler.Serve(path)
}
