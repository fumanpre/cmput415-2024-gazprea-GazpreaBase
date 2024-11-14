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
    std::any visit(std::shared_ptr<AST> t) override;
    std::any visitIF(std::shared_ptr<AST> t) override;
    std::any visitLOOP(std::shared_ptr<AST> t) override;
    std::any visitVAR_DECL(std::shared_ptr<AST> t) override;
    std::any visitASSIGN(std::shared_ptr<AST> t) override;
    std::any visitID(std::shared_ptr<AST> t) override;
    std::any visitINT(std::shared_ptr<AST> t) override;
    std::any visitBOOL(std::shared_ptr<AST> t) override;
    std::any visitREAL(std::shared_ptr<AST> t) override;
    std::any visitCHAR(std::shared_ptr<AST> t) override;
    std::any visitTUPLE(std::shared_ptr<AST> t) override;

};
#endif