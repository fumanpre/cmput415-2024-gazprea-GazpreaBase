#include "AstWalker.h"
#include "GazpreaParser.h"

using namespace gazprea;

std::any AstWalker::visitFILE(std::shared_ptr<AST> t)
{
    return visitChildren(t);
}

std::any AstWalker::visit(std::shared_ptr<AST> t)
{
    if ( t->isNil() ) {
        return visitChildren(t);
    } else {
        switch ( t->getNodeType() ) {
            case GazpreaParser::FILE:
                return visitFILE(t);

            case GazpreaParser::IF:
                return visitIF(t);
                
            case GazpreaParser::VAR_DECL:
                return visitVAR_DECL(t);
                
            case GazpreaParser::ASSIGN:
                return visitASSIGN(t);
                
            case GazpreaParser::ID:
                return visitID(t);
                
            case GazpreaParser::INT:
                return visitINT(t);

            case GazpreaParser::REAL:
                return visitREAL(t);
            
            case GazpreaParser::CHAR:
                return visitCHAR(t);
            
            case GazpreaParser::TUPLE:
                return visitTUPLE(t);
            
            case GazpreaParser::BOOL:
                return visitBOOL(t);

            case GazpreaParser::TUPLE_TYPE:
                return visitTUPLE_TYPE(t);
            
            case GazpreaParser::TUPLE_FIELD:
                return visitTUPLE_FIELD(t);

            case GazpreaParser::TUPLE_ACCESS:
                return visitTUPLE_ACCESS(t);

            case GazpreaParser::MULTI_ASSIGN:
                return visitMULTI_ASSIGN(t);

            case GazpreaParser::DOT_REAL:
                return visitDOT_REAL(t);

            case GazpreaParser::SCIENTIFIC_REAL:
                return visitSCIENTIFIC_REAL(t);

            case GazpreaParser::UNARY_MINUS:
                return visitUNARY_MINUS(t);

            case GazpreaParser::BOOLEAN_NOT:
                return visitBOOLEAN_NOT(t);

            case GazpreaParser::EXPONENT:
                return visitEXPONENT(t);

            case GazpreaParser::MULT:
                return visitMULT(t);

            case GazpreaParser::DIV:
                return visitDIV(t);

            case GazpreaParser::REM:
                return visitREM(t);

            case GazpreaParser::ADD:
                return visitADD(t);

            case GazpreaParser::SUB:
                return visitSUB(t);

            case GazpreaParser::EQUALS:
                return visitEQUALS(t);

            case GazpreaParser::NOTEQUALS:
                return visitNOTEQUALS(t);

            case GazpreaParser::LESS:
                return visitLESS(t);

            case GazpreaParser::GREATER:
                return visitGREATER(t);

            case GazpreaParser::LESSEQUAL:
                return visitLESSEQUAL(t);

            case GazpreaParser::GREATEREQUAL:
                return visitGREATEREQUAL(t);

            case GazpreaParser::BOOLEAN_AND:
                return visitBOOLEAN_AND(t);

            case GazpreaParser::BOOLEAN_OR:
                return visitBOOLEAN_OR(t);

            case GazpreaParser::BOOLEAN_XOR:
                return visitBOOLEAN_XOR(t);
            
            case GazpreaParser::TYPE_CAST_EXPR:
                return visitTYPE_CAST_EXPR(t);
            
            case GazpreaParser::TUPLE_LITERAL:
                return visitTUPLE_LITERAL(t);
                
            case GazpreaParser::TYPEDEF:
                return visitTYPEDEF(t);

            case GazpreaParser::OUTPUTSTREAM:
                return visitOUTPUTSTREAM(t);

            case GazpreaParser::BREAK:
                return visitBREAK(t);

            case GazpreaParser::CONTINUE:
                return visitCONTINUE(t);

            case GazpreaParser::RETURN:
                return visitRETURN(t);

            case GazpreaParser::BLOCK:
                return visitBLOCK(t);

            case GazpreaParser::INPUTSTREAM:
                return visitINPUTSTREAM(t);

            case GazpreaParser::INF_LOOP:
                return visitINF_LOOP(t);

            case GazpreaParser::WHILE_LOOP:
                return visitWHILE_LOOP(t);

            case GazpreaParser::DO_WHILE_LOOP:
                return visitDO_WHILE_LOOP(t);

            case GazpreaParser::FUNC_DECL:
                return visitFUNC_DECL(t);

            case GazpreaParser::FUNC_DEF_PARAM:
                return visitFUNC_DEF_PARAM(t);

            case GazpreaParser::FUNC_DEF_EXPR_RETURN:
                return visitFUNC_DEF_EXPR_RETURN(t);

            case GazpreaParser::FUNC_DEF_PARAMETER_LIST:
                return visitFUNC_DEF_PARAMETER_LIST(t);

            case GazpreaParser::FUNC_DEF_BLOCK_RETURN:
                return visitFUNC_DEF_BLOCK_RETURN(t);

            case GazpreaParser::PROC_DECL_PARAM:
                return visitPROC_DECL_PARAM(t);

            case GazpreaParser::PROC_DEF_PARAM:
                return visitPROC_DEF_PARAM(t);

            case GazpreaParser::PROC_DECL:
                return visitPROC_DECL(t);

            case GazpreaParser::PROC_DEF_PARAMETER_LIST:
                return visitPROC_DEF_PARAMETER_LIST(t);

            case GazpreaParser::CALLSTAT:
                return visitCALLSTAT(t);

            case GazpreaParser::ARG_LIST:
                return visitARG_LIST(t);

            default: // The other nodes we don't care about just have their children return visited
                return visitChildren(t);
        }
    }
}
std::any AstWalker::visitChildren(std::shared_ptr<AST> t) {
    for ( auto child : t->children ) visit(child);
    return std::make_any<int>(0);
}

std::any AstWalker::visitIF(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitLOOP(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitVAR_DECL(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitASSIGN(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitID(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitINT(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitBOOL(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitREAL(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitCHAR(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitTUPLE(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitTUPLE_TYPE(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitTUPLE_FIELD(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitTUPLE_ACCESS(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitMULTI_ASSIGN(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitDOT_REAL(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitSCIENTIFIC_REAL(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitUNARY_MINUS(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitBOOLEAN_NOT(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitEXPONENT(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitMULT(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitDIV(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitREM(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitADD(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitSUB(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitEQUALS(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitNOTEQUALS(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitLESS(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitGREATER(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitLESSEQUAL(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitGREATEREQUAL(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitBOOLEAN_AND(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitBOOLEAN_OR(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitBOOLEAN_XOR(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitTYPE_CAST_EXPR(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitTUPLE_LITERAL(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitTYPEDEF(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitOUTPUTSTREAM(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitBREAK(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitCONTINUE(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitRETURN(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitBLOCK(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitINPUTSTREAM(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitINF_LOOP(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitWHILE_LOOP(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitDO_WHILE_LOOP(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitFUNC_DECL(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitFUNC_DEF_EXPR_RETURN(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitFUNC_DEF_PARAMETER_LIST(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitFUNC_DEF_BLOCK_RETURN(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitPROC_DECL_PARAM(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitPROC_DEF_PARAM(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitPROC_DECL(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitPROC_DEF_PARAMETER_LIST(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitCALLSTAT(std::shared_ptr<AST> t){
    return visitChildren(t);
}

std::any AstWalker::visitARG_LIST(std::shared_ptr<AST> t){
    return visitChildren(t);
}