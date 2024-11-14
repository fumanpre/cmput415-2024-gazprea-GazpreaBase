#include "AstWalker.h"
#include "GazpreaParser.h"

std::any AstWalker::visit(std::shared_ptr<AST> t){
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
    for ( auto child : t->children ) return visit(child);
}

std::any visitIF(std::shared_ptr<AST> t){
    return 0;
}

std::any visitLOOP(std::shared_ptr<AST> t){
    return 0;
}

std::any visitVAR_DECL(std::shared_ptr<AST> t){
    return 0;
}

std::any visitASSIGN(std::shared_ptr<AST> t){
    return 0;
}

std::any visitID(std::shared_ptr<AST> t){
    return 0;
}

std::any visitINT(std::shared_ptr<AST> t){
    return 0;
}

std::any visitBOOL(std::shared_ptr<AST> t){
    return 0;
}

std::any visitREAL(std::shared_ptr<AST> t){
    return 0;
}

std::any visitCHAR(std::shared_ptr<AST> t){
    return 0;
}

std::any visitTUPLE(std::shared_ptr<AST> t){
    return 0;
}

std::any visitTUPLE_TYPE(std::shared_ptr<AST> t){
    return 0;
}

std::any visitTUPLE_FIELD(std::shared_ptr<AST> t){
    return 0;
}

std::any visitTUPLE_ACCESS(std::shared_ptr<AST> t){
    return 0;
}

std::any visitMULTI_ASSIGN(std::shared_ptr<AST> t){
    return 0;
}

std::any visitDOT_REAL(std::shared_ptr<AST> t){
    return 0;
}

std::any visitSCIENTIFIC_REAL(std::shared_ptr<AST> t){
    return 0;
}

std::any visitUNARY_MINUS(std::shared_ptr<AST> t){
    return 0;
}

std::any visitBOOLEAN_NOT(std::shared_ptr<AST> t){
    return 0;
}

std::any visitEXPONENT(std::shared_ptr<AST> t){
    return 0;
}

std::any visitMULT(std::shared_ptr<AST> t){
    return 0;
}

std::any visitDIV(std::shared_ptr<AST> t){
    return 0;
}

std::any visitREM(std::shared_ptr<AST> t){
    return 0;
}

std::any visitADD(std::shared_ptr<AST> t){
    return 0;
}

std::any visitSUB(std::shared_ptr<AST> t){
    return 0;
}

std::any visitEQUALS(std::shared_ptr<AST> t){
    return 0;
}

std::any visitNOTEQUALS(std::shared_ptr<AST> t){
    return 0;
}

std::any visitLESS(std::shared_ptr<AST> t){
    return 0;
}

std::any visitGREATER(std::shared_ptr<AST> t){
    return 0;
}

std::any visitLESSEQUAL(std::shared_ptr<AST> t){
    return 0;
}

std::any visitGREATEREQUAL(std::shared_ptr<AST> t){
    return 0;
}

std::any visitBOOLEAN_AND(std::shared_ptr<AST> t){
    return 0;
}

std::any visitBOOLEAN_OR(std::shared_ptr<AST> t){
    return 0;
}

std::any visitBOOLEAN_XOR(std::shared_ptr<AST> t){
    return 0;
}

std::any visitTYPE_CAST_EXPR(std::shared_ptr<AST> t){
    return 0;
}

std::any visitTUPLE_LITERAL(std::shared_ptr<AST> t){
    return 0;
}

std::any visitTYPEDEF(std::shared_ptr<AST> t){
    return 0;
}

std::any visitOUTPUTSTREAM(std::shared_ptr<AST> t){
    return 0;
}

std::any visitBREAK(std::shared_ptr<AST> t){
    return 0;
}

std::any visitCONTINUE(std::shared_ptr<AST> t){
    return 0;
}

std::any visitRETURN(std::shared_ptr<AST> t){
    return 0;
}

std::any visitBLOCK(std::shared_ptr<AST> t){
    return 0;
}

std::any visitINPUTSTREAM(std::shared_ptr<AST> t){
    return 0;
}

std::any visitINF_LOOP(std::shared_ptr<AST> t){
    return 0;
}

std::any visitWHILE_LOOP(std::shared_ptr<AST> t){
    return 0;
}

std::any visitDO_WHILE_LOOP(std::shared_ptr<AST> t){
    return 0;
}

std::any visitFUNC_DECL(std::shared_ptr<AST> t){
    return 0;
}

std::any visitFUNC_DEF_EXPR_RETURN(std::shared_ptr<AST> t){
    return 0;
}

std::any visitFUNC_DEF_PARAMETER_LIST(std::shared_ptr<AST> t){
    return 0;
}

std::any visitFUNC_DEF_BLOCK_RETURN(std::shared_ptr<AST> t){
    return 0;
}

std::any visitPROC_DECL_PARAM(std::shared_ptr<AST> t){
    return 0;
}

std::any visitPROC_DEF_PARAM(std::shared_ptr<AST> t){
    return 0;
}

std::any visitPROC_DECL(std::shared_ptr<AST> t){
    return 0;
}

std::any visitPROC_DEF_PARAMETER_LIST(std::shared_ptr<AST> t){
    return 0;
}

std::any visitCALLSTAT(std::shared_ptr<AST> t){
    return 0;
}

std::any visitARG_LIST(std::shared_ptr<AST> t){
    return 0;
}
