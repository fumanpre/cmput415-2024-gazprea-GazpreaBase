#include "ScopeResolver.h"
#include "GazpreaParser.h"

using namespace gazprea;

ScopeResolver::ScopeResolver() : symtab(), currentScope(symtab.globals) {}

std::any ScopeResolver::visitFile(std::shared_ptr<AST> t)
{
    t->scope = std::static_pointer_cast<BaseScope>(symtab.globals); // root node holds the global scope
    return AstWalker::visitFile();
}

Qualifier ScopeResolver::resolveQualifier(std::shared_ptr<AST> t){
    if(t->isNil() || t->token->getText() == "var"){
        return VAR;
    }
    return CONST;
}

std::shared_ptr<Type> ScopeResolver::resolveType(std::shared_ptr<AST> t){
    std::shared_ptr<Type> tsym;
    // TO DO add cases for vectors and matrices
    switch ( t->getNodeType() ) {
        case GazpreaParser::TUPLE_TYPE:
            break;
        default:
            tsym = symtab.globals->resolveType(t->token->getText());
    }
    return tsym;
}

std::any ScopeResolver::visitVAR_DECL(std::shared_ptr<AST> t)
{
    visit(t->children[3]); // visit expr node first to resolve the children
    std::shared_ptr<AST> q = t->children[0]; // qualifier
    std::shared_ptr<AST> ty = t->children[1]; // type
    std::shared_ptr<AST> id = t->children[2];
    Qualifier qual = resolveQualifier(q);
    std::shared_ptr<Type> type = resolveType(ty);
    std::shared_ptr<VariableSymbol> vs = std::make_shared<VariableSymbol>(id->token->getText(), type, qual);
    currentScope->define(vs);
    // make the id node hold the symbol ptr
    id->sym = vs;
    return 0;
    
}

