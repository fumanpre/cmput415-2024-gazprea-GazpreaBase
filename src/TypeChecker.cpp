#include "TypeChecker.h"

using namespace gazprea;

std::any TypeChecker::visitFile(std::shared_ptr<AST> t)
{
    TypeChecker::globalScope = t->scope;

}

Qualifier TypeChecker::resolveQualifier(std::shared_ptr<AST> t){
    if(t->isNil() || t->token->getText() == "var"){
        return VAR;
    }
    return CONST;
}

std::any TypeChecker::visitVAR_DECL(std::shared_ptr<AST> t, std::shared_ptr<Scope> currentScope){
    std::shared_ptr<AST> q = t->children[0]; // qualifier
    std::shared_ptr<AST> expr = t->children[3]; // expression
    Qualifier qual = resolveQualifier(q);

    // if global variable is not constant
    if (currentScope->getEnclosingScope() == nullptr)
    {
        if (qual != Qualifier::CONST)
        {
            // throw an error (Global Error)
        }
    }

    std::shared_ptr<Type> exprType = visit(expr); // visit expr node first to get the type of expression;

    // Inferred type (if type == nullptr)
    std::shared_ptr<AST> ty = t->children[1];

    if (ty == nullptr)
    {
        std::shared_ptr<VariableSymbol> vs;
        
        if (expr->getNodeType() != GazpreaParser::TUPLE_TYPE)
        {   
            vs = std::make_shared<VariableSymbol>(id->token->getText(), exprType, qual);
        }
        else
        {   
           // TO DO what is type is a tuple
        }

        currentScope->define(vs);
        t->sym = vs;
        
        return 0;
    }   
    // if (type != nullptr)
    else {
        auto left_side_type = t->sym->type->ty;
        auto right_side_type = expr->sym->type->ty;

        if (left_side_type != right_side_type)
        {
            // skip automatic promotion
            if (left_side_type == TypeEnum::REAL && right_side_type == TypeEnum::INT)
            {
                return 0;
            }
            
            // throw an error (TYPE ERROR)
        }  
    }

    return 0;
}

std::any TypeChecker::visitASSIGN(std::shared_ptr<AST> t){
    visitChildren(t);

    auto id = t->children[0];
    auto expr = t->children[1];

    auto variable = std::dynamic_pointer_cast<VariableSymbol>(id->sym);

    if (variable->qual == Qualifier::CONST)
    {
        // throw an error (??)
    }

    if (variable->type->ty != expr->sym->type->ty)
    {
        // skip automatic promotion
        if (variable->type->ty == TypeEnum::REAL && expr->sym->type->ty == TypeEnum::INT)
        {
            return 0;
        }
        
        // throw an error (TYPE ERROR)
    }

    return 0;
}

std::any TypeChecker::visitIF(std::shared_ptr<AST> t){
    auto expression = t->children[0];
    auto if_stat = t->children[1];

    if (expression->sym->type->ty != TypeEnum::BOOL)
    {
        // throw a error (Type ERROR)
    }

    auto it = begin(t->children);
    it++; // skip the first expr
    for (; it != end(t->children); it++)
    {
        visit(*it); // visit the children that is the statements inside the block
    }

    return 0;
}

std::any TypeChecker::visitINF_LOOP(std::shared_ptr<AST> t){
    visitChildren(t);
    // how to check if there is a break inside
    return 0;
}

std::any TypeChecker::visitWHILE_LOOP(std::shared_ptr<AST> t){ // same as if
    auto expression = t->children[0];

    if (expression->sym->type->ty != TypeEnum::BOOL)
    {
        // throw an error (Type ERROR)
    }

    // visit statements inside the while loop
    visit(t->children[1]);   

    return 0;
}

std::any TypeChecker::visitDO_WHILE_LOOP(std::shared_ptr<AST> t){ // same as if
    visit(t->children[0]); // visit the expression first
    
    auto expression = t->children[1];

    if (expression->sym->type->ty != TypeEnum::BOOL)
    {
        // throw an error (Type Error)
    }
}

std::any TypeChecker::visitID(std::shared_ptr<AST> t){

    if(t->sym->type->getName() == "integer"){
        t->sym->type->ty = INT;
    }
    else if(t->sym->type->getName() == "boolean"){
        t->sym->type->ty = BOOL;
    }
    else if(t->sym->type->getName() == "character"){
        t->sym->type->ty = CHAR;
    }
    else if(t->sym->type->getName() == "real"){
        t->sym->type->ty = REAL;
    }
    else if(t->sym->type->getName() == "tuple"){
        t->sym->type->ty = TUPLE;
    }

    // else throw error
}

std::any TypeChecker::visitINT(std::shared_ptr<AST> t){
    t->exprType = num;
}

std::any TypeChecker::visitArithOp(std::shared_ptr<AST> t){
    visitChildren(t);
    std::shared_ptr<AST> expr1 = t->children[0];
    std::shared_ptr<AST> expr2 = t->children[1];
    if( expr1->exprType == vec || expr2->exprType == vec ){
        t->exprType = vec;
    }
    else{
        t->exprType = num;
    }
}

std::any TypeChecker::visitBoolOp(std::shared_ptr<AST> t){
    visitChildren(t);
    std::shared_ptr<AST> expr1 = t->children[0];
    std::shared_ptr<AST> expr2 = t->children[1];
    if( expr1->exprType == vec || expr2->exprType == vec ){
        t->exprType = vec;
    }
    else{
        t->exprType = num;
    }
}

