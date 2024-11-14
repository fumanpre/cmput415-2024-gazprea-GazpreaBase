#include "Symbols.h"

Symbol::Symbol(std::string name) : Symbol(name, nullptr) {}
Symbol::Symbol(std::string name, std::shared_ptr<Type> type) : name(name), type(type) {}

std::string Symbol::getName() { return name; }
std::string Symbol::getFullName() { return name+scope->getScopeName(); }
std::string Symbol::toString() {
    if (type != nullptr) return '<' + getFullName() + ":" + type->getName() + '>';
    return getFullName();
}

Symbol::~Symbol() {}

Type::~Type() {}

BuiltInTypeSymbol::BuiltInTypeSymbol(std::string name) : Symbol(name) {}

std::string BuiltInTypeSymbol::getName() {
    return Symbol::getName();
}

VariableSymbol::VariableSymbol(std::string name, std::shared_ptr<Type> type) : Symbol(name, type), addrHolder(nullptr) {}

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

TupleSymbol::TupleSymbol( std::string name, std::shared_ptr<Scope> enclosingScope) 
    : ScopedSymbol(name, enclosingScope) {}

std::shared_ptr<Symbol> TupleSymbol::resolve(const std::string &name) {
    if (fields.count(name) == 1) {
        return fields.at(name);
    }

    // if not here, check any enclosing scope
    if ( getEnclosingScope() != nullptr ) {
        return getEnclosingScope()->resolve(name);
    }

    return nullptr; // not found
}

std::shared_ptr<Symbol> TupleSymbol::resolveMember(const std::string &name) {
    if (fields.count(name) == 1) {
        return fields.at(name);
    }
    // No enclosing scope for a Tuple
    return nullptr;
}

void TupleSymbol::define(std::shared_ptr<Symbol> sym) {
    fields.emplace(sym->name, sym);
    sym->scope = shared_from_this();
}

std::string TupleSymbol::getScopeName() {
    return name;
}

std::string TupleSymbol::getName() {
    return name;
}

std::string TupleSymbol::toString() {
    std::stringstream str;
    str << "Tuple " << Symbol::toString() << " {" << std::endl;
    for (auto const& f : fields) {
        std::shared_ptr<Symbol> sym = f.second;
        str << "\t" << sym->toString() << std::endl;
    }
    str << "}";
    return str.str();
}