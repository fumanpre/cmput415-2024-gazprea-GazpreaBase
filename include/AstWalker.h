#ifndef ASTWALKER_H
#define ASTWALKER_H

#include "Ast.h"
#include "GazpreaBaseVisitor.h"

using namespace gazprea;

class AstWalker {
public:
    virtual std::any visit(std::shared_ptr<AST> t);
    virtual std::any visitChildren(std::shared_ptr<AST> t);
    virtual std::any visitIF(std::shared_ptr<AST> t);
    virtual std::any visitLOOP(std::shared_ptr<AST> t);
    virtual std::any visitVAR_DECL(std::shared_ptr<AST> t);
    virtual std::any visitASSIGN(std::shared_ptr<AST> t);
    virtual std::any visitID(std::shared_ptr<AST> t);
    virtual std::any visitINT(std::shared_ptr<AST> t);
    virtual std::any visitBOOL(std::shared_ptr<AST> t);
    virtual std::any visitREAL(std::shared_ptr<AST> t);
    virtual std::any visitCHAR(std::shared_ptr<AST> t);
    virtual std::any visitTUPLE(std::shared_ptr<AST> t);

};
#endif