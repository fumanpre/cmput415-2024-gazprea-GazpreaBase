#ifndef ASTWALKER_H
#define ASTWALKER_H

#include "Ast.h"

class AstWalker {
public:
    virtual std::any visitFILE(std::shared_ptr<AST> t);
    virtual std::any visit(std::shared_ptr<AST> t);
    virtual std::any visitChildren(std::shared_ptr<AST> t);
    virtual std::any visitVAR_DECL(std::shared_ptr<AST> t);
    virtual std::any visitASSIGN(std::shared_ptr<AST> t);
    virtual std::any visitID(std::shared_ptr<AST> t);
    virtual std::any visitINT(std::shared_ptr<AST> t);
    virtual std::any visitBOOL(std::shared_ptr<AST> t);
    virtual std::any visitREAL(std::shared_ptr<AST> t);
    virtual std::any visitCHAR(std::shared_ptr<AST> t);
    virtual std::any visitTUPLE_TYPE(std::shared_ptr<AST> t);
    virtual std::any visitTUPLE_FIELD(std::shared_ptr<AST> t);
    virtual std::any visitTUPLE_ACCESS(std::shared_ptr<AST> t);
    virtual std::any visitMULTI_ASSIGN(std::shared_ptr<AST> t);
    virtual std::any visitDOT_REAL(std::shared_ptr<AST> t);
    virtual std::any visitSCIENTIFIC_REAL(std::shared_ptr<AST> t);
    virtual std::any visitUNARY_MINUS(std::shared_ptr<AST> t);
    virtual std::any visitBOOLEAN_NOT(std::shared_ptr<AST> t);
    virtual std::any visitEXPONENT(std::shared_ptr<AST> t);
    virtual std::any visitMULT(std::shared_ptr<AST> t);
    virtual std::any visitDIV(std::shared_ptr<AST> t);
    virtual std::any visitREM(std::shared_ptr<AST> t);
    virtual std::any visitADD(std::shared_ptr<AST> t);
    virtual std::any visitSUB(std::shared_ptr<AST> t);
    virtual std::any visitEQUALS(std::shared_ptr<AST> t);
    virtual std::any visitNOTEQUALS(std::shared_ptr<AST> t);
    virtual std::any visitLESS(std::shared_ptr<AST> t);
    virtual std::any visitGREATER(std::shared_ptr<AST> t);
    virtual std::any visitLESSEQUAL(std::shared_ptr<AST> t);
    virtual std::any visitGREATEREQUAL(std::shared_ptr<AST> t);
    virtual std::any visitBOOLEAN_AND(std::shared_ptr<AST> t);
    virtual std::any visitBOOLEAN_OR(std::shared_ptr<AST> t);
    virtual std::any visitBOOLEAN_XOR(std::shared_ptr<AST> t);
    virtual std::any visitTYPE_CAST_EXPR(std::shared_ptr<AST> t);
    virtual std::any visitTUPLE_LITERAL(std::shared_ptr<AST> t);
    virtual std::any visitTYPEDEF(std::shared_ptr<AST> t);
    virtual std::any visitOUTPUTSTREAM(std::shared_ptr<AST> t);
    virtual std::any visitIF(std::shared_ptr<AST> t);
    virtual std::any visitBREAK(std::shared_ptr<AST> t);
    virtual std::any visitCONTINUE(std::shared_ptr<AST> t);
    virtual std::any visitRETURN(std::shared_ptr<AST> t);
    virtual std::any visitBLOCK(std::shared_ptr<AST> t);
    virtual std::any visitINPUTSTREAM(std::shared_ptr<AST> t);
    virtual std::any visitINF_LOOP(std::shared_ptr<AST> t);
    virtual std::any visitWHILE_LOOP(std::shared_ptr<AST> t);
    virtual std::any visitDO_WHILE_LOOP(std::shared_ptr<AST> t);
    virtual std::any visitFUNC_DECL(std::shared_ptr<AST> t);
    virtual std::any visitFUNC_DEF_EXPR_RETURN(std::shared_ptr<AST> t);
    virtual std::any visitFUNC_DEF_PARAMETER_LIST(std::shared_ptr<AST> t);
    virtual std::any visitFUNC_DEF_BLOCK_RETURN(std::shared_ptr<AST> t);
    virtual std::any visitPROC_DECL_PARAM(std::shared_ptr<AST> t);
    virtual std::any visitPROC_DEF_PARAM(std::shared_ptr<AST> t);
    virtual std::any visitPROC_DECL(std::shared_ptr<AST> t);
    virtual std::any visitPROC_DEF_PARAMETER_LIST(std::shared_ptr<AST> t);
    virtual std::any visitCALLSTAT(std::shared_ptr<AST> t);
    virtual std::any visitARG_LIST(std::shared_ptr<AST> t);

};
#endif