#include "Ast.h"
#include "AstBuildingVisitor.h"

// Each visitor returns the AST node and hence the subtree

std::any AstBuildingVisitor::visitFile(GazpreaParser::FileContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::FILE);
    for ( auto stat : ctx->stat() ) {
        t->addChild(visit(stat));
    }
    return t;
}


// ^(VAR_DECL qualifier type ID expr)
std::any AstBuildingVisitor::visitDecl(GazpreaParser::DeclContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::VAR_DECL);
    if( ctx->qualifier() != nullptr ){
        // qualifier present
        t->addChild(visit(ctx->qualifier()));
    }
    else{
        //qualifier absent add nil node
        t->addChild(std::make_shared<AST>());
    }
    t->addChild(visit(ctx->fixedSizeType())); // add type node
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol())); // add ID node
    if( ctx->expr() != nullptr ){
        // expr present
        t->addChild(visit(ctx->expr()));
    }
    else{
        //expr absent add nil node
        t->addChild(std::make_shared<AST>());
    }
    return t;
}

// ^(VAR_DECL qualifier type ID expr)
std::any AstBuildingVisitor::visitInferredDecl(GazpreaParser::InferredDeclContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::VAR_DECL);
    t->addChild(visit(ctx->qualifier())); // add qualifier node
    t->addChild(std::make_shared<AST>()); // add nil node for type
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol())); // add ID node
    t->addChild(visit(ctx->expr())); // add expr node
    return t;
}

// ^(VAR_DECL qualifier type ID expr)
std::any AstBuildingVisitor::visitInferredDecl(GazpreaParser::InferredDeclContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::VAR_DECL);
    if( ctx->qualifier() != nullptr ){
        // qualifier present
        t->addChild(visit(ctx->qualifier()));
    }
    else{
        //qualifier absent add nil node
        t->addChild(std::make_shared<AST>());
    }
    t->addChild(visit(ctx->varSizeType())); // add type node
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol())); // add ID node
    t->addChild(visit(ctx->expr())); // add expr node
    return t;
}

// ^(QUALIFIER)     eg. ^(VAR)
std::any AstBuildingVisitor::visitQualifier(GazpreaParser::QualifierContext *ctx){
    // make AST node from the first token in this context
    return std::make_shared<AST>(ctx->getStart()); 
}


// ^(TYPE)         eg. ^(INTEGER)
std::any AstBuildingVisitor::visitType(GazpreaParser::TypeContext *ctx){
    // make AST node from the first token in this context
    return std::make_shared<AST>(ctx->getStart()->getSymbol()); 
}

// ^(TUPLE_TYPE TUPLE_FIELD+)
std::any AstBuildingVisitor::visitTupleType(GazpreaParser::TupleTypeContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::TUPLE_TYPE);
    for ( auto tf : ctx->tupleField() ) {
        t->addChild(visit(tf));
    }
    return t;
}

// ^(TUPLE_FIELD TYPE ID)
std::any AstBuildingVisitor::visitTupleField(GazpreaParser::TupleFieldContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::TUPLE_FIELD);
    t->addChild(visit(ctx->tupleFieldType())); // add type node
    if( ctx->ID() != nullptr ){
        t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol())); // add ID node
    }
    else{
        //ID absent add nil node
        t->addChild(std::make_shared<AST>());
    }
    return t;
}

// START NEXT ITERATION HERE


// ^(ASSIGN ID EXPR)
std::any AstBuildingVisitor::visitAssignmentStat(GazpreaParser::AssignmentStatContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::ASSIGN);
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol()));
    t->addChild(visit(ctx->expr()));
    return t;
}
// ^(IF expr stat* )
std::any AstBuildingVisitor::visitConditionalStat(GazpreaParser::ConditionalStatContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::IF);
    t->addChild(visit(ctx->expr()));
    for ( auto stat : ctx->stat() ) {
        t->addChild(visit(stat));
    }
    return t;
}
// ^(LOOP expr stat* )
std::any AstBuildingVisitor::visitLoopStat(GazpreaParser::LoopStatContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::LOOP);
    t->addChild(visit(ctx->expr()));
    for ( auto stat : ctx->stat() ) {
        t->addChild(visit(stat));
    }
    return t;
}
// ^(PRINT expr)
std::any AstBuildingVisitor::visitPrintStat(GazpreaParser::PrintStatContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::PRINT);
    t->addChild(visit(ctx->expr()));
    return t;
}

std::any AstBuildingVisitor::visitType(GazpreaParser::TypeContext *ctx){
    // make AST node from the first token in this context
    return std::make_shared<AST>(ctx->getStart()); 
}

// ^(MULT expr expr) | ^(DIV expr expr)
std::any AstBuildingVisitor::visitMultDivExpr(GazpreaParser::MultDivExprContext *ctx){
    std::shared_ptr<AST> t(nullptr);
    if (ctx->op->getText() == "*")
        t = std::make_shared<AST>(GazpreaParser::MULT);
    else
        t = std::make_shared<AST>(GazpreaParser::DIV);
    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));
    return t;
}
// ^(ADD expr expr) | ^(SUB expr expr)
std::any AstBuildingVisitor::visitAddSubExpr(GazpreaParser::AddSubExprContext *ctx){
    std::shared_ptr<AST> t(nullptr);
    if (ctx->op->getText() == "+")
        t = std::make_shared<AST>(GazpreaParser::ADD);
    else
        t = std::make_shared<AST>(GazpreaParser::SUB);
    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));
    return t;
}
// ^(LESS expr expr) | ^(GREAT expr expr)
std::any AstBuildingVisitor::visitLessGreatExpr(GazpreaParser::LessGreatExprContext *ctx){
    std::shared_ptr<AST> t(nullptr);
    if (ctx->op->getText() == "<")
        t = std::make_shared<AST>(GazpreaParser::LESS);
    else
        t = std::make_shared<AST>(GazpreaParser::GREAT);
    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));
    return t;
}
// ^(EQUALS expr expr) | ^(NOTEQUALS expr expr)
std::any AstBuildingVisitor::visitEqNotEqExpr(GazpreaParser::EqNotEqExprContext *ctx){
    std::shared_ptr<AST> t(nullptr);
    if (ctx->op->getText() == "==")
        t = std::make_shared<AST>(GazpreaParser::EQUALS);
    else
        t = std::make_shared<AST>(GazpreaParser::NOTEQUALS);
    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));
    return t;
}


// DO TILL HERE
std::any AstBuildingVisitor::visitParenthesisExpr(GazpreaParser::ParenthesisExprContext *ctx){
    return visit(ctx->expr());
}

std::any AstBuildingVisitor::visitIdExpr(GazpreaParser::IdExprContext *ctx){
    return std::make_shared<AST>(ctx->ID()->getSymbol()); 
}

std::any AstBuildingVisitor::visitIntLiteralExpr(GazpreaParser::IntLiteralExprContext *ctx){
    return std::make_shared<AST>(ctx->INT()->getSymbol());
}

std::any AstBuildingVisitor::visitBoolLiteralExpr(GazpreaParser::BoolLiteralExprContext *ctx){
    return std::make_shared<AST>(ctx->BOOL()->getSymbol());
}

std::any AstBuildingVisitor::visitCharLiteralExpr(GazpreaParser::CharLiteralExprContext *ctx){
    return std::make_shared<AST>(ctx->CHAR()->getSymbol());
}

std::any AstBuildingVisitor::visitPreDotReal(GazpreaParser::PreDotRealContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::DOT_REAL);
    t->addChild(std::make_shared<AST>(ctx->INT(0)->getSymbol()));
    if( ctx->INT(1) != nullptr){
        t->addChild(std::make_shared<AST>(ctx->INT(1)->getSymbol()));
    }
    return t;
}

std::any AstBuildingVisitor::visitPostDotReal(GazpreaParser::PostDotRealContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::DOT_REAL);
    if(ctx->getStart() == GazpreaParser::DOT){
        t->addChild(std::make_shared<AST>());
        t->addChild(std::make_shared<AST>(ctx->INT()->getSymbol()));
    }
    else{
        t->addChild(std::make_shared<AST>(ctx->INT(0)->getSymbol()));
        t->addChild(std::make_shared<AST>(ctx->INT(1)->getSymbol()));
    }
    return t;
}

std::any AstBuildingVisitor::visitIntEReal(GazpreaParser::IntERealContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::SCIENTIFIC_REAL);
    t->addChild(std::make_shared<AST>(ctx->INT(0)->getSymbol()));
    t->addChild(std::make_shared<AST>(ctx->INT(1)->getSymbol()));
    return t;
}

std::any AstBuildingVisitor::visitRealEReal(GazpreaParser::RealERealContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::SCIENTIFIC_REAL);
    t->addChild(visit(ctx->real()));
    t->addChild(std::make_shared<AST>(ctx->INT()->getSymbol()));
    return t;
}