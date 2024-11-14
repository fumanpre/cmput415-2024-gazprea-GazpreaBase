#ifndef SCOPERESOLVER_H
#define SCOPERESOLVER_H

#include "AstWalker.h"
#include "SymbolTable.h"
#include "Scoping.h"
#include "Symbols.h"

class ScopeResolver : public AstWalker{
    SymbolTable symtab;
    std::shared_ptr<Scope> currentScope;
    std::shared_ptr<Type> resolveType(std::shared_ptr<AST> t);
    public:
    ScopeResolver();
    void visit(std::shared_ptr<AST> t) override;
};
#endif