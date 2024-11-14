#include "Scoping.h"
#include <sstream>
Scope::~Scope() {}
int BaseScope::numScopes = 0; // used to distinguish between scope names
BaseScope::BaseScope(std::shared_ptr<Scope> enclosingScope) : enclosingScope(enclosingScope), scopeNum(BaseScope::numScopes) {
    BaseScope::numScopes++;
}

std::shared_ptr<Symbol> BaseScope::resolve(const std::string &name) {
    auto find_s = symbols.find(name);
    if ( find_s != symbols.end() ) return find_s->second;
    // if not here, check any enclosing scope
    if ( enclosingScope != nullptr ) return enclosingScope->resolve(name);
    return nullptr; // not found
}

void BaseScope::define(std::shared_ptr<Symbol> sym) {
    symbols.emplace(sym->name, sym);
    sym->scope = shared_from_this(); // track the scope in each symbol
}

std::shared_ptr<Scope> BaseScope::getEnclosingScope() {
    return enclosingScope;
}

std::string BaseScope::toString() {
    std::stringstream str;
    str << "{";
    for (auto iter = symbols.begin(); iter != symbols.end(); iter++) {
        std::shared_ptr<Symbol> sym = iter->second;
        if ( iter != symbols.begin() ) str << ", ";
        str << sym->toString();
    }
    str << "}";
    return str.str();
}

GlobalScope::GlobalScope() : BaseScope(nullptr) {}

std::string GlobalScope::getScopeName() {
    return "global"+std::to_string(scopeNum);
}

LocalScope::LocalScope(std::shared_ptr<Scope> parent) : BaseScope(parent) {}

std::string LocalScope::getScopeName() {
    return "local"+std::to_string(scopeNum);
}