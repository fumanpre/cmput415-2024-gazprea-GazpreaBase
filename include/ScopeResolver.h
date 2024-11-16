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
    std::any visitFile(std::shared_ptr<AST> t) override;
    std::any visitVAR_DECL(std::shared_ptr<AST> t) override;
    std::any visitID(std::shared_ptr<AST> t); override;
    std::any visitTUPLE_TYPE(std::shared_ptr<AST> t);
    std::any visitTUPLE_FIELD(std::shared_ptr<AST> t);
    std::any visitTUPLE_ACCESS(std::shared_ptr<AST> t);

};
#endif