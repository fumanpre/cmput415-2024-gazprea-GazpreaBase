#include "ScopeResolver.h"
#include "CompileTimeException.h"
#include <vector>

ScopeResolver::ScopeResolver() : symtab(), currentScope(symtab.globals) {}

std::shared_ptr<Type> ScopeResolver::resolveType(std::shared_ptr<AST> t) {
    std::shared_ptr<Type> tsym;
    tsym = std::dynamic_pointer_cast<Type>(symtab.globals->resolve(t->token->getText()));
    return tsym;
}


std::any ScopeResolver::visit(std::shared_ptr<AST> t){
    if ( t->isNil() ) {
        t->scope = std::static_pointer_cast<BaseScope>(symtab.globals); // root node holds the global scope
    }
    AstWalker::visit(t); // call the parent's visit
}


std::any ScopeResolver::visitVAR_DECL(std::shared_ptr<AST> t){
    std::shared_ptr<AST> qualifier = t->children[0];    // qualifier
    std::shared_ptr<AST> ty = t->children[1];   // type
    std::shared_ptr<AST> id = t->children[2];   // variable name
    std::shared_ptr<AST> expr = t->children[3]; // expression

    if (expr != nullptr)
    {
        visit(expr);
    }

    // if id name is already in use by other declaration
    

    // making a variable symbol
    std::shared_ptr<VariableSymbol> vs = std::make_shared<VariableSymbol>(id->token->getText(), ?, ?, ?);

    // Define that variable symbol in current scope
    currentScope->define(vs);

    // make the id node hold the varibale symbol ptr
    id->sym = vs;

    // delete the qualifier and type node from the ast
    t->deleteFirstChild();  // for qualifier
    t->deleteFirstChild();  // for type

    // etc.,.
    // make the id node hold the scope the id is in


    return 0;
}

std::any ScopeResolver::visitIF(std::shared_ptr<AST> t){
    return 0;
}

std::any ScopeResolver::visitLOOP(std::shared_ptr<AST> t){
    return 0;
}

//  ^(ID '=' expr)
std::any ScopeResolver::visitASSIGN(std::shared_ptr<AST> t){
    visitChildren(t);

    return 0;
}

std::any ScopeResolver::visitID(std::shared_ptr<AST> t){
    std::shared_ptr<Symbol> s = currentScope->resolve(t->token->getText());

    // If s is nullptr ( not resolved throw exception s)
    if (s == nullptr){
        throw SymbolError( ?? (line_number), "Undefined symbol is referenced.");
    }
    

    t->sym = std::dynamic_pointer_cast<VariableSymbol> (s);

    return 0;
}

std::any ScopeResolver::visitINT(std::shared_ptr<AST> t){
    return 0;
}

std::any ScopeResolver::visitBOOL(std::shared_ptr<AST> t){
    return 0;
}

std::any ScopeResolver::visitCHAR(std::shared_ptr<AST> t){
    return 0;
}

std::any ScopeResolver::visitTUPLE(std::shared_ptr<AST> t){
    return 0;
}

std::any ScopeResolver::visitREAL(std::shared_ptr<AST> t){
    return 0;
}
