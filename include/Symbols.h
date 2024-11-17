#ifndef SYMBOL_H
#define SYMBOL_H

#include <string>
#include <memory>

#include "mlir/IR/Value.h"

#include "Scoping.h"

class Scope; // forward declaration of Scope to resolve circular dependency


enum TypeEnum{
    INT,
    CHAR,
    BOOL,
    REAL,
    TUPLE,
    VECTOR,
    MATRIX
}

class Type {
public:
    TypeEnum ty;
    Type( TypeEnum t ) : ty(t) {}
    virtual std::string getName() = 0;
    virtual ~Type();
};

class Symbol { // A generic programming language symbol
public:
    std::string name;               // All symbols at least have a name
    std::shared_ptr<Type> type;
    std::shared_ptr<Scope> scope;   // All symbols know what scope contains them.

    Symbol(std::string name);
    Symbol(std::shared_ptr<Type> type); // for unnamed symbols with just types (in tuples)
    Symbol(std::string name, std::shared_ptr<Type> type);
    virtual std::string getName();
    virtual std::string getFullName();

    virtual std::string toString();
    virtual ~Symbol();
};

class BuiltInTypeSymbol : public Symbol, public Type {
public:
    BuiltInTypeSymbol(std::string name, TypeEnum t);
    std::string getName();
};

enum Qualifier{
    CONST,
    VAR
};

class VariableSymbol : public Symbol {
public:
    // mlir::Value addr;
    Qualifier qual;
    VariableSymbol(std::string name, std::shared_ptr<Type> type);
    VariableSymbol(std::string name, std::shared_ptr<Type> type, Qualifier q);
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

class TupleType : public Type {

    public:
    TupleType() : Type(TUPLE) {}
    std::map<std::string, int> filedNameToIndexMap;
    std::vector<std::shared_ptr<Symbol>> fields;

}

class TupleSymbol : public ScopedSymbol{ // To update
public:
    Qualifier qual;

    TupleSymbol(std::string name, std::shared_ptr<Type> t, std::shared_ptr<Scope> enclosingScope, Qualifier q);
    void define(std::shared_ptr<Symbol> sym);

    virtual std::shared_ptr<Symbol> resolve(const std::string &name);
    virtual std::shared_ptr<Symbol> resolveMember(const std::string &name);
    virtual std::shared_ptr<Symbol> resolveMember(int index);

    virtual std::string getScopeName();
    virtual std::string getName();

    std::string toString();
};

#endif