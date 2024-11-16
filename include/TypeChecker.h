#ifndef TYPECHECKER_H
#define TYPECHECKER_H
#include "Ast.h"
#include "AstWalker.h"

class TypeChecker : public AstWalker{
    std::shared_ptr<Scope> globalScope;
    public:
    std::any visitFile(std::shared_ptr<AST> t) override;
    std::any visitIF(std::shared_ptr<AST> t) override;
    std::any visitLOOP(std::shared_ptr<AST> t) override;
    std::any visitVAR_DECL(std::shared_ptr<AST> t) override;
    std::any visitASSIGN(std::shared_ptr<AST> t) override;
    std::any visitPRINT(std::shared_ptr<AST> t) override;
    std::any visitID(std::shared_ptr<AST> t) override;
    std::any visitINT(std::shared_ptr<AST> t) override;
    std::any visitArithOp(std::shared_ptr<AST> t) override; //could be opened to diff operators
    std::any visitBoolOp(std::shared_ptr<AST> t) override;
    std::any visitGeneratorExpr(std::shared_ptr<AST> t) override;
    std::any visitFilterExpr(std::shared_ptr<AST> t) override;
    std::any visitRangeExpr(std::shared_ptr<AST> t) override;
    std::any visitIndexExpr(std::shared_ptr<AST> t) override;

};
#endif