#ifndef SYMBOL_H
#define SYMBOL_H

#include <string>
#include <memory>

#include "mlir/IR/Value.h"

#include "Scoping.h"

class Scope; // forward declaration of Scope to resolve circular dependency

class Type {
public:
    virtual std::string getName() = 0;
    virtual ~Type();
};

class Symbol { // A generic programming language symbol
public:
    std::string name;               // All symbols at least have a name
    std::shared_ptr<Type> type;
    std::shared_ptr<Scope> scope;   // All symbols know what scope contains them.

    Symbol(std::string name);
    Symbol(std::string name, std::shared_ptr<Type> type);
    virtual std::string getName();
    virtual std::string getFullName();

    virtual std::string toString();
    virtual ~Symbol();
};

class BuiltInTypeSymbol : public Symbol, public Type {
public:
    BuiltInTypeSymbol(std::string name);
    std::string getName();
};

class TupleTypeSymbol : public Symbol, public Type {
public:
    BuiltInTypeSymbol(std::string name) : Symbol(name) {}
    std::string getName() { return Symbol::getName(); }
};

class VariableSymbol : public Symbol {
public:
    // mlir::Value addr;
    VariableSymbol(std::string name, std::shared_ptr<Type> type);
};

class ScopedSymbol : public Symbol, public Scope {
 public:
	std::shared_ptr<Scope> enclosingScope; // nullptr if global (outermost) scope
    ScopedSymbol(std::string name, std::shared_ptr<Scope> enClosingScope);
    ScopedSymbol(std::string name, std::shared_ptr<Type> type,
                 std::shared_ptr<Scope> enClosingScope);

    /** Look up name in this scope or in enclosing scope if not here */
    virtual std::shared_ptr<Symbol> resolve(const std::string &name) = 0;

    virtual std::shared_ptr<Scope> getEnclosingScope();

    virtual ~ScopedSymbol();
};

class MethodSymbol : public ScopedSymbol {
private:
    std::vector<std::shared_ptr<Symbol>> orderedArgs; 
public:
    MethodSymbol(std::string name, std::shared_ptr<Type> retType, std::shared_ptr<Scope> enclosingScope);

    virtual std::shared_ptr<Symbol> resolve(const std::string &name);
    void define(std::shared_ptr<Symbol> sym);

    std::string getScopeName();
    std::string toString();
};

class TupleSymbol : public ScopedSymbol, public Type {
public:
	std::map<std::string, std::shared_ptr<Symbol>> fields;

    TupleSymbol(std::string name, std::shared_ptr<Scope> enclosingScope);

    void define(std::shared_ptr<Symbol> sym);

    virtual std::shared_ptr<Symbol> resolve(const std::string &name);
    virtual std::shared_ptr<Symbol> resolveMember(const std::string &name);

    virtual std::string getScopeName();
    virtual std::string getName();

    std::string toString();
};

#endif