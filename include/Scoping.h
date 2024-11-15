#ifndef SCOPE_H
#define SCOPE_H

#include <string>
#include <memory>
#include <map>

#include "Symbols.h"

class Symbol; // forward declaration of Symbol to resolve circular dependency

class Scope : public std::enable_shared_from_this<Scope> {
public:
    virtual std::string getScopeName() = 0;

    /** Where to look next for symbols */
    virtual std::shared_ptr<Scope> getEnclosingScope() = 0;

    /** Define a symbol in the current scope */
    virtual void define(std::shared_ptr<Symbol> sym) = 0;

    /** Look up name in this scope or in enclosing scope if not here */
    virtual std::shared_ptr<Symbol> resolve(const std::string &name) = 0;

	virtual std::string toString() = 0;
    virtual ~Scope();
};
class BaseScope : public Scope {
    static int numScopes;
public:
    const int scopeNum; 
	std::shared_ptr<Scope> enclosingScope; // nullptr if global (outermost) scope
	std::map<std::string, std::shared_ptr<Symbol>> symbols;

    BaseScope(std::shared_ptr<Scope> enclosingScope);
    std::shared_ptr<Symbol> resolve(const std::string &name) override;
	void define(std::shared_ptr<Symbol> sym) override;
    std::shared_ptr<Scope> getEnclosingScope() override;

	virtual std::string toString() override;
};

class GlobalScope : public BaseScope {
public:
    std::map<std::string, std::shared_ptr<Type>> types;
    std::shared_ptr<Type> resolveType(const std::string &name);
	void defineType(std::shared_ptr<Type> typ);
    void defineType(const std::string &name, std::shared_ptr<Type> typ);
    GlobalScope();
    std::string getScopeName() override;
};
class LocalScope : public BaseScope {
public:
    LocalScope(std::shared_ptr<Scope> parent);
    std::string getScopeName() override;
};
#endif