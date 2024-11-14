#include "AstWalker.h"
#include "GazpreaParser.h"

std::any AstWalker::visit(std::shared_ptr<AST> t){
    if ( t->isNil() ) {
        return visitChildren(t);
    } else {
        switch ( t->getNodeType() ) {
            case GazpreaParser::IF:
                return visitIF(t);
                
            case GazpreaParser::LOOP:
                return visitLOOP(t);
                
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