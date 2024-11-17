#include "TypeChecker.h"
#include "compileTimeExceptions.h"

using namespace gazprea;

std::any TypeChecker::visitFile(std::shared_ptr<AST> t)
{
    globalScope = t->scope;

}

Qualifier TypeChecker::resolveQualifier(std::shared_ptr<AST> t){
    if(t->isNil() || t->token->getText() == "var"){
        return VAR;
    }
    return CONST;
}

// current scope is not found in ast.h 
std::any TypeChecker::visitVAR_DECL(std::shared_ptr<AST> t){
    std::shared_ptr<AST> q = t->children[0]; // qualifier
    std::shared_ptr<AST> expr = t->children[3]; // expression
    Qualifier qual = resolveQualifier(q);

    // if variable declaration is global declaration
    if (currentScope->getEnclosingScope() == nullptr)
    {
        // if global variable is not constant
        if (qual != Qualifier::CONST)
        {
            throw GlobalError(t->token->getLine(), "Global variable is not constant.");
        }
        // if global variable is without expression
        if (expr == nullptr)
        {
            throw GlobalError(t->token->getLine(), "Global variable is without expression.");
        }

        // TO DO if in global variable, the expression is function call or something related
        // throw an Global Error
    }

    std::shared_ptr<Type> exprType = visit(expr); // visit expr node first to get the type of expression;

    // Inferred type (if type == nullptr)
    std::shared_ptr<AST> ty = t->children[1];

    if (ty == nullptr)
    {
        std::shared_ptr<Symbol> symbol;
        
        if (expr->getNodeType() != GazpreaParser::TUPLE_TYPE)
        {   
            symbol = std::make_shared<VariableSymbol>(id->token->getText(), exprType, qual);
        }
        else
        {   
           // TO DO what is type is a tuple
           symbol = std::make_shared<TupleSymbol>(id->token->getText(), type, qual);
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
            throw TypeError(t->token->getLine(), "Type Error, mismatching type while variable declaration.");
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
        // throw an error (AssignError)
        throw AssignError(t->token->getLine(), "Assign Error, assigning value to a const variable.");
    }

    if (variable->type->ty != expr->sym->type->ty)
    {
        // skip automatic promotion
        if (variable->type->ty == TypeEnum::REAL && expr->sym->type->ty == TypeEnum::INT)
        {
            return 0;
        }
        
        // throw an error (TYPE ERROR)
        throw TypeError(t->token->getLine(), "Type Error, mismatching type while variable assignment.");
        
    }

    return 0;
}

std::any TypeChecker::visitIF(std::shared_ptr<AST> t){
    auto expression = t->children[0];
    auto if_stat = t->children[1];

    if (expression->sym->type->ty != TypeEnum::BOOL)
    {
        // throw a error (Type ERROR)
        throw TypeError(t->token->getLine(), "Type Error, If expression type is not boolean.");

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
    // Flag to indicate the beginning of loop
    loopStack.push(1);
    inf_loop_count += 1;

    visitChildren(t);
    
    loopStack.pop();
    inf_loop_count -= 1;

    // if there is a break inside a loop
    // throw an error if there is no break statement
    if (breakCount != 0)
    {
        throw StatementError(t->token->getLine(), "Infinite loop does not contain break statement.");
    }
    else breakCount -= 1;

    return 0;
}

std::any TypeChecker::visitWHILE_LOOP(std::shared_ptr<AST> t){ // same as if
    // Flag to indicate the beginning of loop
    loopStack.push(1);

    auto expression = t->children[0];

    if (expression->sym->type->ty != TypeEnum::BOOL)
    {
        // throw an error (Type ERROR)
        throw TypeError(t->token->getLine(), "Type Error, While loop expression type is not boolean.");

    }

    // visit statements inside the while loop
    visit(t->children[1]);   

    loopStack.pop();

    return 0;
}

std::any TypeChecker::visitDO_WHILE_LOOP(std::shared_ptr<AST> t){ // same as if
    // Flag to indicate the beginning of loop
    loopStack.push(1);

    visit(t->children[0]); // visit the expression first
    
    auto expression = t->children[1];

    if (expression->sym->type->ty != TypeEnum::BOOL)
    {
        // throw an error (Type Error)
        throw TypeError(t->token->getLine(), "Type Error, do while loop expression type is not boolean.");

    }

    loopStack.pop();
}

std::any TypeChecker::visitID(std::shared_ptr<AST> t){

    if(t->sym->type->getName() == "integer"){
        t->sym->type->ty = TypeEnum::INT;
    }
    else if(t->sym->type->getName() == "boolean"){
        t->sym->type->ty = TypeEnum::BOOL;
    }
    else if(t->sym->type->getName() == "character"){
        t->sym->type->ty = TypeEnum::CHAR;
    }
    else if(t->sym->type->getName() == "real"){
        t->sym->type->ty = TypeEnum::REAL;
    }
    else if(t->sym->type->getName() == "tuple"){
        t->sym->type->ty = TypeEnum::TUPLE;
    }

    // else throw error
}

std::any TypeChecker::visitINT(std::shared_ptr<AST> t)
    auto int_type = globalScope->resolveType("integer");
    t->sym->type->ty = TypeEnum::INT;
    
    return int_type;


std::any TypeChecker::visitCHAR(std::shared_ptr<AST> t){
    auto char_type = globalScope->resolveType("character");
    t->sym->type->ty = TypeEnum::CHAR;

    return char_type;
}

std::any TypeChecker::visitBOOL(std::shared_ptr<AST> t){
    auto bool_type = globalScope->resolveType("boolean");
    t->sym->type->ty = TypeEnum::BOOL;

    return bool_type;
}

std::any TypeChecker::visitREAL(std::shared_ptr<AST> t){
    auto real_type = globalScope->resolveType("real");
    t->sym->type->ty = TypeEnum::REAL;

    return real_type;
}


// what if there is no symbol associated with sym in ast for expression nodes (like multiply, add, etc.,.)
std::any TypeChecker::visitEXPR(std::shared_ptr<AST> t){
    auto children_size = t->children.size();

    if (children_size == 2)
    {
        auto left_child = t->children[0];
        auto right_child = t->children[1];

        // visit both expression first
        visit(left_child);
        visit(right_child);

        auto left_child_type = left_child->sym->type->ty;
        auto right_child_type = right_child->sym->type->ty;
        // expression type 
        auto expression_type = t->sym->type->ty;

        if (t->token->getText() == '+' || t->token->getText() == '-' || t->token->getText() == '*')
        {
            if ( left_child_type == TypeEnum::INT)
            {
                switch (right_child_type) {
                    case TypeEnum::INT:
                        expression_type = TypeEnum::INT;  // INT + INT or INT - INT -> INT
                        break;
                    case TypeEnum::REAL:
                        expression_type = TypeEnum::REAL; // INT + REAL or INT - REAL -> REAL
                        break;
                    default:
                        // throw type error
                        throw TypeError(t->token->GetLine(), "Addition, Subtraction or Multiplication between incompatible types.");
                }
            }
            else if (left_child_type == TypeEnum::REAL)
            {
                switch (right_child_type) {
                    case TypeEnum::INT:
                    case TypeEnum::REAL:
                        expression_type = TypeEnum::REAL; // REAL + INT/REAL or REAL - INT/REAL -> REAL
                        break;
                    default:
                        // throw type error
                        throw TypeError(t->token->GetLine(), "Addition, Subtraction or Multiplication between incompatible types.");
                }
            }
            else
            {
                // throw a type error
                throw TypeError(t->token->GetLine(), "Addition, Subtraction or Multiplication between incompatible types.");
            }
        }
        else if (t->token->getText() == '/')
        {
            if (left_child_type == TypeEnum::INT)
            {
                switch (right_child_type) {
                    case TypeEnum::INT:
                        expression_type = TypeEnum::INT; // INT + INT or INT - INT -> INT
                        break;
                    case TypeEnum::REAL:
                        expression_type = TypeEnum::REAL; // INT + REAL or INT - REAL -> REAL
                        break;
                    default:
                        //  throw type error
                        throw TypeError(t->token->GetLine(), "Division between incompatible types.");
                }
            }
            else if (left_child_type == TypeEnum::REAL)
            {
                switch (right_child_type) {
                    case TypeEnum::INT:
                    case TypeEnum::REAL:
                        expression_type = TypeEnum::REAL;  // INT + CHAR or INT - CHAR -> INT
                        break;
                    default:
                        // throw type error
                        throw TypeError(t->token->GetLine(), "Division between incompatible types.");
                        
                }
            }
            else{
                // throw an type error
                throw TypeError(t->token->GetLine(), "Division between incompatible types.");
            }
        }
        else if (t->token->getText() == '<' || t->token->getText() == "<=" || t->token->getText() == '>' || t->token->getText() == ">=")
        {
            if (left_child_type == TypeEnum::INT || left_child_type == TypeEnum::REAL)
            {
                switch (right_child_type) {
                    case TypeEnum::INT:   
                    case TypeEnum::REAL:
                        expression_type = TypeEnum::BOOL;
                        break;
                    default:
                        // throw an type error
                        throw TypeError(t->token->GetLine(), " <, >, >=, <= between incompatible types.");
                }
            }
            else{
                // throw an typeerror
                throw TypeError(t->token->GetLine(), " <, >, >=, <= between incompatible types.");
            }
        }
        else if (t->token->getText() == "==" || t->token->getText() == "!=")
        {
            if (left_child_type == TypeEnum::INT || left_child_type == TypeEnum::REAL)
            {
                switch (right_child_type) {
                    case TypeEnum::INT:   
                    case TypeEnum::REAL:
                        expression_type = TypeEnum::BOOL;
                        break;
                    default:
                        // throw an type error
                        throw TypeError(t->token->GetLine(), " ==, != between incompatible types.");
                }
            }
            else if (left_child_type == TypeEnum::BOOL)
            {
                switch (right_child_type) {
                    case TypeEnum::BOOL:
                        expression_type = TypeEnum::BOOL
                        break;
                    default:
                        // throw an type error
                        throw TypeError(t->token->GetLine(), " ==, != between incompatible types.");
                }
            }
            else if (left_child_type == TypeEnum::CHAR)
            {
                switch (right_child_type) {
                    case TypeEnum::CHAR:
                        expression_type = TypeEnum::BOOL;
                        break;
                    default:
                        // throw an type error
                        throw TypeError(t->token->GetLine(), " ==, != between incompatible types.");
                }
            }
            else{
                // throw an error
                throw TypeError(t->token->GetLine(), " ==, != between incompatible types.");
            }
        }
         else if (t->token->getText() == '%')
        {
            if (left_child_type == TypeEnum::INT)
            {
                switch (right_child_type) {
                    case TypeEnum::INT:
                    case TypeEnum::REAL:
                        expression_type = TypeEnum::INT;
                        break;
                    default:
                        // throw an error
                        throw TypeError(t->token->GetLine(), " % between incompatible types.");
                }
            }
            else if (left_child_type == TypeEnum::REAL)
            {
                switch (right_child_type) {
                    case TypeEnum::INT:
                    case TypeEnum::REAL:
                        expression_type = TypeEnum::REAL;
                        break;
                    default:
                        // throw an error
                        throw TypeError(t->token->GetLine(), " % between incompatible types.");
                }
            }
            else{
                // throw an error
                throw TypeError(t->token->GetLine(), " % between incompatible types.");
            }
        }
        else if (t->token->getText() == '^')
        {
            if (left_child_type == TypeEnum::INT)
            {
                switch (right_child_type) {
                    case TypeEnum::INT:
                        expression_type = TypeEnum::INT;
                        break;
                    case TypeEnum::REAL:
                        expression_type = TypeEnum::REAL;
                        break;
                    default:
                        // throw an error
                        throw TypeError(t->token->GetLine(), " ^ between incompatible types.");
                }
            }
            else if (left_child_type == TypeEnum::REAL)
            {
                switch (right_child_type) {
                    case TypeEnum::INT:
                    case TypeEnum::REAL:
                        expression_type = TypeEnum::REAL;
                        break;
                    default:
                        // throw an error
                        throw TypeError(t->token->GetLine(), " ^ between incompatible types.");
                }
            }
            else{
                // throw an error
                throw TypeError(t->token->GetLine(), " ^ between incompatible types.");
            }
        }
        else if (t->token->getText() == "or" || t->token->getText() == "xor")
        {
            if (left_child_type == TypeEnum::BOOL)
            {
                switch (right_child_type) {
                    case TypeEnum::BOOL:
                        expression_type = TypeEnum::BOOL;
                        break;
                    default:
                        // thow an type error
                        throw TypeError(t->token->GetLine(), " or, xor between incompatible types.");
                }
            }
            else{
                // throw an error
                throw TypeError(t->token->GetLine(), " or, xor between incompatible types.");
            }
        }
        // // how about dot operation
        // else if (expr->GetText() == '.')
        // {
        //     if (left_type->GetType() == Type::GazpreaTypes::INT || left_type->GetType() == Type::GazpreaTypes::BOOL || left_type->GetType() == Type::GazpreaTypes::CHAR)
        //     {
        //         switch (right_type->GetType()) {
        //             case Type::GazpreaTypes::INT:
        //             case Type::GazpreaTypes::CHAR:
        //             case Type::GazpreaTypes::BOOL:
        //                 current_node->SetTypeReference(int_type); 
        //                 break;
        //             case Type::GazpreaTypes::REAL:
        //                 current_node->SetTypeReference(float_type); 
        //                 break;
        //             default:
        //                 throw std::runtime_error("Invalid operation with INT and this type.");
        //         }
        //     }
        //     else if (left_type->GetType() == Type::GazpreaTypes::REAL)
        //     {
        //         switch (right_type->GetType()) {
        //             case Type::GazpreaTypes::INT:
        //             case Type::GazpreaTypes::CHAR:
        //             case Type::GazpreaTypes::BOOL:
        //             case Type::GazpreaTypes::REAL:
        //                 current_node->SetTypeReference(float_type); 
        //                 break;
        //             default:
        //                 throw std::runtime_error("Invalid operation with INT and this type.");
        //         }
        //     }
        //     else{
        //         // throw an error
        //     }
        // }
    }
    else if (children_size == 1){
        auto expression_type = t->sym->type->ty;
        auto child_type = t->children[0]->sym->type->ty;

        if (t->token->getText() == "not")
        {
            if (child_type == TypeEnum::BOOL)
            {
                expression_type = TypeEnum::BOOL;
            }
            else{
                // throw type error
                throw TypeError(t->token->GetLine(), " Unary not operation is applied to incompatible type.");
            }
        }
        // effectively ignore the "+"
        else if (t->token->getText() == '-')
        {
            if (child_type == TypeEnum::INT) 
            {
                expression_type = TypeEnum::INT;
            }
            else if (child_type == TypeEnum::REAL)
            {
                expression_type = TypeEnum::REAL;
            }
            else{
                // throw type error
                throw TypeError(t->token->GetLine(), " Unary Addition OR Subtraction is applied to incompatible type.");
            }
        }
    }

    // resolve the type for the expression as It is being used in VAR_DECL
    switch (t->sym->type->ty)
    {
        case TypeEnum::INT:
            return visitINT(t);
        case TypeEnum::REAL:
            return visitREAL(t);
        case TypeEnum::CHAR:
            return visitCHAR(t);
        case TypeEnum::BOOL:
            return visitBOOL(t);
        default:
            return 0;
    }
}

std::any TypeChecker::visitBREAK(std::shared_ptr<AST> t){
    breakCount += 1;

    if (loopStack.size() == 0)
    {
        // throw an error
        throw StatementError(t->token->getLine(), "Break is not in enclosed loop.");
    }

    return 0;
}

std::any TypeChecker::visitCONTINUE(std::shared_ptr<AST> t){
    if (loopStack.size() == 0)
    {
        // throw an error
        throw StatementError(t->token->getLine(), "Continue is not in enclosed loop.");
    }

    return 0;
}

// TO DO just providing enum is enought or resolved type also
std::any TypeChecker::visitTYPE_CAST_EXPR(std::shared_ptr<AST> t){
    auto to_type = t->children[0];
    auto expr = t->children[1];

    visit(expr);

    auto to_type_resolved = globalScope->resolveType(to_type);

    if (to_type_resolved == nullptr)
    {
        // throw an error
        throw TypeError(t->token->getLine(), "Expression can not be casted to given type (TUPLE).");
    }

    // using pointer to type in order to compare types
    auto int_type = visitINT(to_type);
    auto bool_type = visitBOOL(to_type);
    auto char_type = visitCHAR(to_type);
    auto real_type = visitREAL(to_type);

    auto expr_type = expr->sym->type->ty;

    if (to_type_resolved == char_type)
    {
        switch (expr_type)
        {
        case TypeEnum::BOOL:
        case TypeEnum::INT:
        case TypeEnum::CHAR:
            expr_type = TypeEnum::CHAR;
            break;
        
        default:
            // throw an error
            throw TypeError(t->token->getLine(), "In casting, Expression can not be casted to given type (TUPLE).");
        }
    }
    else if (to_type_resolved == bool_type)
    {
        switch (expr_type)
        {
        case TypeEnum::CHAR:
        case TypeEnum::INT:
        case TypeEnum::BOOL:
            expr_type = TypeEnum::BOOL;
            break;
        
        default:
            // throw an error
            throw TypeError(t->token->getLine(), "In casting, Expression can not be casted to given type (TUPLE).");
        }
    }
    else if (to_type_resolved == int_type)
    {
        switch (expr_type)
        {
        case TypeEnum::CHAR:
        case TypeEnum::BOOL:
        case TypeEnum::INT:
        case TypeEnum::REAL:
            expr_type = TypeEnum::INT;
            break;
        
        default:
            // throw an error
            throw TypeError(t->token->getLine(), "In casting, Expression can not be casted to given type (TUPLE).");
        }
    }
    else if (to_type_resolved == real_type)
    {
        switch (expr_type)
        {
        case TypeEnum::CHAR:
        case TypeEnum::BOOL:
        case TypeEnum::INT:
        case TypeEnum::REAL:
            expr_type = TypeEnum::REAL;
            break;
        
        default:
            // throw an error
            throw TypeError(t->token->getLine(), "In casting, Expression can not be casted to given type (TUPLE).");
        }
    }

    return 0;
}
