#include "Symbols.h"

Symbol::Symbol(std::string name) : Symbol(name, nullptr) {}
Symbol::Symbol(std::shared_ptr<Type> type) : name(""), type(type) {}
Symbol::Symbol(std::string name, std::shared_ptr<Type> type) : name(name), type(type) {}

std::string Symbol::getName() { return name; }
std::string Symbol::getFullName() { return name+scope->getScopeName(); }
std::string Symbol::toString() {
    if (type != nullptr) return '<' + getFullName() + ":" + type->getName() + '>';
    return getFullName();
}

Symbol::~Symbol() {}

Type::~Type() {}

BuiltInTypeSymbol::BuiltInTypeSymbol(std::string name, TypeEnum t) : Symbol(name), Type(t) {}

std::string BuiltInTypeSymbol::getName() {
    return Symbol::getName();
}

VariableSymbol::VariableSymbol(std::string name, std::shared_ptr<Type> type) : Symbol(name, type), qual(VAR) {}
VariableSymbol::VariableSymbol(std::string name, std::shared_ptr<Type> type, Qualifier q) : Symbol(name, type), qual(q) {}

ScopedSymbol::ScopedSymbol(std::string name, std::shared_ptr<Scope> scope)
    : Symbol(name), enclosingScope(scope) {}
ScopedSymbol::ScopedSymbol(std::string name, std::shared_ptr<Type> type,
                           std::shared_ptr<Scope> scope)
    : Symbol(name, type), enclosingScope(scope) {}

std::shared_ptr<Scope> ScopedSymbol::getEnclosingScope() {
    return enclosingScope;
}

ScopedSymbol::~ScopedSymbol() {}


MethodSymbol::MethodSymbol( std::string name, std::shared_ptr<Type> retType,
                            std::shared_ptr<Scope> enclosingScope) 
    : ScopedSymbol(name, retType, enclosingScope) {}

std::shared_ptr<Symbol> MethodSymbol::resolve(const std::string &name) {
    for ( auto sym : orderedArgs ) {
        if ( sym->getName() == name ) {
            return sym;
        }
    }
    // if not here, check any enclosing scope
    if ( getEnclosingScope() != nullptr ) {
        return getEnclosingScope()->resolve(name);
    }
    return nullptr; // not found
}

void MethodSymbol::define(std::shared_ptr<Symbol> sym) {
    orderedArgs.push_back(sym);
    sym->scope = shared_from_this();
}

std::string MethodSymbol::getScopeName() { return name; }

std::string MethodSymbol::toString() {
    std::stringstream str;
    str << "method" << Symbol::toString() << ":{";
    for (auto iter = orderedArgs.begin(); iter != orderedArgs.end(); iter++) {
        std::shared_ptr<Symbol> sym = *iter;
        if ( iter != orderedArgs.begin() ) str << ", ";
        str << sym->toString();
    }
    str << "}";
    return str.str();
}

TupleSymbol(std::string name, std::shared_ptr<Type> t, std::shared_ptr<Scope> enclosingScope, Qualifier q)
    : ScopedSymbol(name, t, enclosingScope), qual(q) {}

std::shared_ptr<Symbol> TupleSymbol::resolve(const std::string &name) {
    // Won't be used so is empty
}

std::shared_ptr<Symbol> TupleSymbol::resolveMember(const std::string &name) {
    if (type->fieldNameToIndex.count(name) == 1) {
        return resolveMember(type->fieldNameToIndex.at(name));
    }
    // No enclosing scope for a Tuple
    return nullptr;
}

std::shared_ptr<Symbol> TupleSymbol::resolveMember(int index) {
    if (index<=(type->fields).size()) {
        return (type->fields)[index-1];
    }
    // No enclosing scope for a Tuple
    return nullptr;
}

void TupleSymbol::define(std::shared_ptr<Symbol> sym) {} // won't be used

std::string TupleSymbol::getScopeName() {
    return name;
}

std::string TupleSymbol::getName() {
    return name;
}

std::string TupleSymbol::toString() {
    std::stringstream str;
    str << "Tuple " << Symbol::toString() << " {" << std::endl;
    for (auto const& f : type->fields) {
        str << "\t" << f->toString() << std::endl;
    }
    str << "}";
    return str.str();
}