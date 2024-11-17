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
            auto tt = std::make_shared<TupleType>();
            int index = 1;
            for( auto tuple_field : t->children ){
                auto type_child = tuple_field->children[0]; // tuple field type ast node
                auto childType = resolveType(type_child); // shared pointer to type for the field
                auto vs = std::make_shared<VariableSymbol>(childType);
                (tt->fields).push_back(vs);
                auto name_child = tuple_field->children[1];
                if( !name_child.isNil() ){
                    (tt->fieldNameToIndexMap).emplace(name_child->token->getText(), index);
                }
                index++;
            }
            tsym = std::static_pointer_cast<Type>(tt);
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
    if( ty->isNil() )
        return 0; // this case should be handled while type check inferrred dcl
    std::shared_ptr<AST> id = t->children[2];
    Qualifier qual = resolveQualifier(q);
    std::shared_ptr<Type> type = resolveType(ty);
    std::shared_ptr<Symbol> symbol 
    if( ty->getNodeType() == GazpreaParser::TUPLE_TYPE ){
        symbol = std::make_shared<TupleSymbol>(id->token->getText(), type, currentScope, qual);
    }
    else{
        symbol = std::make_shared<VariableSymbol>(id->token->getText(), type, qual);
    }
    currentScope->define(symbol);
    // make the id node hold the symbol ptr
    id->sym = symbol;
    // make the ast node hold currentScope useful for inferred decl
    t->scope = currentScope;

    return 0;
    
}

std::any ScopeResolver::visitID(std::shared_ptr<AST> t){
    std::shared_ptr<Symbol> s = currentScope->resolve(t->token->getText());
    // If s is nullptr ( not resolved throw exception s)
    t->sym = s;
    return 0;
}

std::any ScopeResolver::visitTUPLE_ACCESS(std::shared_ptr<AST> t){
    std::shared_ptr<AST> id = t->children[0];
    visit(id); // resolve the id node
    auto tupleSym = std::dynamic_pointer_cast<TupleSymbol>(id->sym);
    std::shared_ptr<AST> child2 = t->children[1];
    if( child2->getNodeType == GazpreaParser::ID){
        t->sym = tupleSym->resolveMember(child2->token->getText());
    }
    else{
        t->sym = tupleSym->resolveMember(std::stoi(child2->token->getText()));
    }
    return 0;
}


std::any ScopeResolver::visitTYPEDEF(std::shared_ptr<AST> t){
    std::shared_ptr<AST> ty = t->children[0];
    auto tsym = resolveType(ty);
    std::shared_ptr<AST> name = t->children[1];
    symtab.globals->defineType(name->token->getText(), tsym);
    return 0;
}
