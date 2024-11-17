#ifndef TYPECHECKER_H
#define TYPECHECKER_H
#include "Ast.h"
#include "AstWalker.h"

class TypeChecker : public AstWalker{
    std::shared_ptr<GlobalScope> globalScope;
    // Create a stack to store numbers using in loop
    std::stack<int> loopStack;
    int inf_loop_count = 0;
    int breakCount = 0;
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

};
#endif