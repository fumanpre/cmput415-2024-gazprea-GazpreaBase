#include "Ast.h"
#include "AstBuildingVisitor.h"

// Each visitor returns the AST node and hence the subtree

std::any AstBuildingVisitor::visitFile(GazpreaParser::FileContext *ctx){
    std::cout << "Visiting File\n"; // Debug print
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::FILE);
    for ( auto stat : ctx->stat() ) {
        std::cout << "adding stat\n"; // Debug print
        t->addChild(visit(stat));
        std::cout << "fin stat\n"; // Debug print
    }
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitDeclarationStatement(GazpreaParser::DeclarationStatementContext *ctx){
    return visit(ctx->variableDeclaration());
}

std::any AstBuildingVisitor::visitAssignmentStatement(GazpreaParser::AssignmentStatementContext *ctx){
    return visit(ctx->assignment());
}

// ^(VAR_DECL qualifier type ID expr)
std::any AstBuildingVisitor::visitDecl(GazpreaParser::DeclContext *ctx){
    std::cout << "Visiting Decl\n"; // Debug print
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
    std::cout << "ExitDecl\n"; // Debug print
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(VAR_DECL qualifier type ID expr)
std::any AstBuildingVisitor::visitInferredDecl(GazpreaParser::InferredDeclContext *ctx){
    std::cout << "Visiting InferredDecl\n"; // Debug print
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::VAR_DECL);
    t->addChild(visit(ctx->qualifier())); // add qualifier node
    t->addChild(std::make_shared<AST>()); // add nil node for type
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol())); // add ID node
    t->addChild(visit(ctx->expr())); // add expr node
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(VAR_DECL qualifier type ID expr)
std::any AstBuildingVisitor::visitVarSizedDecl(GazpreaParser::VarSizedDeclContext *ctx){
    std::cout << "Visiting VarSizedDecl\n"; // Debug print
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::VAR_DECL);
    if( ctx->qualifier() != nullptr ){
        // qualifier present
        t->addChild(visit(ctx->qualifier()));
    }
    else{
        //qualifier absent add nil node
        t->addChild(std::make_shared<AST>());
    }
    t->addChild(visit(ctx->varSizedType())); // add type node
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol())); // add ID node
    t->addChild(visit(ctx->expr())); // add expr node
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(QUALIFIER)     eg. ^(VAR)
std::any AstBuildingVisitor::visitQualifier(GazpreaParser::QualifierContext *ctx){
    std::cout << "Visiting Qualifier\n"; // Debug print
    // make AST node from the first token in this context
    return std::make_shared<AST>(ctx->getStart()); 
}


// ^(TYPE)         eg. ^(INTEGER)
std::any AstBuildingVisitor::visitType(GazpreaParser::TypeContext *ctx){
    std::cout << "Visiting Type\n"; // Debug print
    // make AST node from the first token in this context
    return std::make_shared<AST>(ctx->getStart()); 
}

// ^(TUPLE_TYPE TUPLE_FIELD+)
std::any AstBuildingVisitor::visitTupleType(GazpreaParser::TupleTypeContext *ctx){
    std::cout << "Visiting TupleType\n"; // Debug print
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::TUPLE_TYPE);
    for ( auto tf : ctx->tupleField() ) {
        t->addChild(visit(tf));
    }
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(TUPLE_FIELD TYPE ID)
std::any AstBuildingVisitor::visitTupleField(GazpreaParser::TupleFieldContext *ctx){
    std::cout << "Visiting TupleField\n"; // Debug print
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::TUPLE_FIELD);
    t->addChild(visit(ctx->tupleFieldType())); // add type node
    if( ctx->ID() != nullptr ){
        t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol())); // add ID node
    }
    else{
        //ID absent add nil node
        t->addChild(std::make_shared<AST>());
    }
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitIdLVal(GazpreaParser::IdLValContext *ctx){
    return std::make_any<std::shared_ptr<AST>>(std::make_shared<AST>(ctx->ID()->getSymbol()));
}

std::any AstBuildingVisitor::visitTupleAccessLVal(GazpreaParser::TupleAccessLValContext *ctx){
    std::shared_ptr<AST> child1 = std::make_shared<AST>(GazpreaParser::TUPLE_ACCESS);
    child1->addChild(std::make_shared<AST>(ctx->ID(0)->getSymbol()));
    if( ctx->ID(1) != nullptr ){
        child1->addChild(std::make_shared<AST>(ctx->ID(1)->getSymbol()));
    }
    else{
        child1->addChild(std::make_shared<AST>(ctx->INT()->getSymbol()));
    }
    return std::make_any<std::shared_ptr<AST>>(child1);
}

std::any AstBuildingVisitor::visitLValAssign(GazpreaParser::LValAssignContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::ASSIGN);
    t->addChild(visit(ctx->lVal())); // add lVal node
    t->addChild(visit(ctx->expr()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitMultiAssign(GazpreaParser::MultiAssignContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::ASSIGN);
    std::shared_ptr<AST> child1 = std::make_shared<AST>(GazpreaParser::MULTI_ASSIGN);
    for ( auto lv : ctx->lVal() ) {
        child1->addChild(visit(lv));
    }
    t->addChild(child1); // add MULTI_ASSIGN node
    t->addChild(visit(ctx->expr()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitParenthesisExpr(GazpreaParser::ParenthesisExprContext *ctx){
    std::cout << "Visiting ParentheisiExpr\n"; // Debug print
    return visit(ctx->expr());
}

std::any AstBuildingVisitor::visitIdExpr(GazpreaParser::IdExprContext *ctx){
    std::cout << "Visiting visitIdExpr\n"; // Debug print
    return std::make_shared<AST>(ctx->ID()->getSymbol()); 
}

std::any AstBuildingVisitor::visitIntLiteralExpr(GazpreaParser::IntLiteralExprContext *ctx){
    std::cout << "Visiting IntLiteralExpr\n"; // Debug print
    return std::make_shared<AST>(ctx->INT()->getSymbol());
}

std::any AstBuildingVisitor::visitBoolLiteralExpr(GazpreaParser::BoolLiteralExprContext *ctx){
    std::cout << "Visiting BoolLiteral\n"; // Debug print
    return std::make_shared<AST>(ctx->BOOL()->getSymbol());
}

std::any AstBuildingVisitor::visitCharLiteralExpr(GazpreaParser::CharLiteralExprContext *ctx){
    std::cout << "Visiting CharLiteral\n"; // Debug print
    return std::make_shared<AST>(ctx->CHAR()->getSymbol());
}
std::any AstBuildingVisitor::visitPostDotReal(GazpreaParser::PostDotRealContext *ctx){
    std::cout << "Visiting PostDotReal\n"; // Debug print
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::DOT_REAL);
    t->addChild(std::make_shared<AST>(ctx->INT(0)->getSymbol()));
    if( ctx->INT(1) != nullptr){
        t->addChild(std::make_shared<AST>(ctx->INT(1)->getSymbol()));
    }
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitPreDotReal(GazpreaParser::PreDotRealContext *ctx){
    std::cout << "Visiting PreDotReal\n"; // Debug print
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::DOT_REAL);
    if(ctx->getStart()->getType() == GazpreaParser::DOT){
        t->addChild(std::make_shared<AST>());
        t->addChild(std::make_shared<AST>(ctx->INT(1)->getSymbol()));
    }
    else{
        t->addChild(std::make_shared<AST>(ctx->INT(0)->getSymbol()));
        t->addChild(std::make_shared<AST>(ctx->INT(1)->getSymbol()));
    }
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitIntEReal(GazpreaParser::IntERealContext *ctx){
    std::cout << "Visiting intEReal\n"; // Debug print
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::SCIENTIFIC_REAL);
    t->addChild(std::make_shared<AST>(ctx->INT(0)->getSymbol()));
    t->addChild(std::make_shared<AST>(ctx->INT(1)->getSymbol()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitRealEReal(GazpreaParser::RealERealContext *ctx){
    std::cout << "Visiting ReaEReal\n"; // Debug print
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::SCIENTIFIC_REAL);
    t->addChild(visit(ctx->real()));
    t->addChild(std::make_shared<AST>(ctx->INT()->getSymbol()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitDotAccessExpr(GazpreaParser::DotAccessExprContext *ctx)
{
    std::shared_ptr<AST> child1 = std::make_shared<AST>(GazpreaParser::TUPLE_ACCESS);
    child1->addChild(std::make_shared<AST>(ctx->ID(0)->getSymbol()));
    if( ctx->ID(1) != nullptr ){
        child1->addChild(std::make_shared<AST>(ctx->ID(1)->getSymbol()));
    }
    else{
        child1->addChild(std::make_shared<AST>(ctx->INT()->getSymbol()));
    }
    return std::make_any<std::shared_ptr<AST>>(child1);
}

std::any AstBuildingVisitor::visitUnaryExpr(GazpreaParser::UnaryExprContext *ctx)
{
    if(ctx->op->getText() == "+"){
        return visit(ctx->expr()); // effectively ignore the plus
    }
    std::shared_ptr<AST> t;
    if(ctx->op->getText() == "-"){
        t = std::make_shared<AST>(GazpreaParser::UNARY_MINUS);
    }
    else{
        t = std::make_shared<AST>(GazpreaParser::BOOLEAN_NOT);
    }
    t->addChild(visit(ctx->expr()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitExponentExpr(GazpreaParser::ExponentExprContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::EXPONENT);
    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitMultDivRemExpr(GazpreaParser::MultDivRemExprContext *ctx)
{
    std::shared_ptr<AST> t;
    if(ctx->op->getText() == "*"){
        t = std::make_shared<AST>(GazpreaParser::MULT);
    }
    else if(ctx->op->getText() == "/"){
        t = std::make_shared<AST>(GazpreaParser::DIV);
    }
    else{
        t = std::make_shared<AST>(GazpreaParser::REM);
    }
    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitAddSubExpr(GazpreaParser::AddSubExprContext *ctx)
{
    std::shared_ptr<AST> t;
    if(ctx->op->getText() == "+"){
        t = std::make_shared<AST>(GazpreaParser::ADD);
    }
    else if(ctx->op->getText() == "-"){
        t = std::make_shared<AST>(GazpreaParser::SUB);
    }
    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitEqNotEqExpr(GazpreaParser::EqNotEqExprContext *ctx)
{
    std::shared_ptr<AST> t;
    if(ctx->op->getText() == "=="){
        t = std::make_shared<AST>(GazpreaParser::EQUALS);
    }
    else if(ctx->op->getText() == "!="){
        t = std::make_shared<AST>(GazpreaParser::NOTEQUALS);
    }
    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitLessGreatExpr(GazpreaParser::LessGreatExprContext *ctx)
{
    std::shared_ptr<AST> t;
    if(ctx->op->getText() == "<"){
        t = std::make_shared<AST>(GazpreaParser::LESS);
    }
    else if(ctx->op->getText() == ">"){
        t = std::make_shared<AST>(GazpreaParser::GREATER);
    }
    else if(ctx->op->getText() == "<="){
        t = std::make_shared<AST>(GazpreaParser::LESSEQUAL);
    }
    else if(ctx->op->getText() == ">="){
        t = std::make_shared<AST>(GazpreaParser::GREATEREQUAL);
    }
    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitBooleanAndExpr(GazpreaParser::BooleanAndExprContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::BOOLEAN_AND);
    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitBooleanOrExpr(GazpreaParser::BooleanOrExprContext *ctx)
{
    std::shared_ptr<AST> t;
    if(ctx->op->getText() == "or"){
        t = std::make_shared<AST>(GazpreaParser::BOOLEAN_OR);
    }
    else if(ctx->op->getText() == "xor"){
        t = std::make_shared<AST>(GazpreaParser::BOOLEAN_XOR);
    }
    t->addChild(visit(ctx->expr(0)));
    t->addChild(visit(ctx->expr(1)));
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(as ALLTYPE EXPR)
std::any AstBuildingVisitor::visitTypeCastExpr(GazpreaParser::TypeCastExprContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::TYPE_CAST_EXPR);
    t->addChild(visit(ctx->allTypes()));
    t->addChild(visit(ctx->expr()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(TUPLELITERAL EXPR+ )
std::any AstBuildingVisitor::visitTupleLiteralExpr(GazpreaParser::TupleLiteralExprContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::TUPLE_LITERAL);
    for ( auto exp : ctx->expr() ) {
        t->addChild(visit(exp));
    }
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(TYPEDEF ALLTYPES ID)
std::any AstBuildingVisitor::visitTypedefStatement(GazpreaParser::TypedefStatementContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::TYPEDEF);
    t->addChild(visit(ctx->allTypes()));
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(OUTPUTSTREAM expr)
std::any AstBuildingVisitor::visitOutputStatement(GazpreaParser::OutputStatementContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::OUTPUTSTREAM);
    t->addChild(visit(ctx->expr()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(IF conditionalexpr ifstat elsestat)
std::any AstBuildingVisitor::visitIfStatement(GazpreaParser::IfStatementContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::IF);
    t->addChild(visit(ctx->expr()));
    t->addChild(visit(ctx->stat(0)));
    if( ctx->ELSE() != nullptr ){
        t->addChild(visit(ctx->stat(1))); // add elseStat
    }
    else{
        t->addChild(std::make_shared<AST>()); // add nil node
    }
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(BREAK)
std::any AstBuildingVisitor::visitBreakStatement(GazpreaParser::BreakStatementContext * ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::BREAK);
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(CONTINUE)
std::any AstBuildingVisitor::visitContinueStatement(GazpreaParser::ContinueStatementContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::CONTINUE);
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(RETURN expr)
std::any AstBuildingVisitor::visitReturnStatement(GazpreaParser::ReturnStatementContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::RETURN);
    if( ctx->expr() != nullptr ){
        t->addChild(visit(ctx->expr()));
    }
    else{
        t->addChild(std::make_shared<AST>());
    }
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(BLOCK stat*)
std::any AstBuildingVisitor::visitBlockStat(GazpreaParser::BlockStatContext * ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::BLOCK);
    for ( auto s : ctx->stat() ) {
        t->addChild(visit(s));
    }
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(INPUTSTREAM LVAL)
std::any AstBuildingVisitor::visitInputStatement(GazpreaParser::InputStatementContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::INPUTSTREAM);
    t->addChild(visit(ctx->lVal()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(INFLOOP stat)
std::any AstBuildingVisitor::visitInfiniteLoop(GazpreaParser::InfiniteLoopContext * ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::INF_LOOP);
    t->addChild(visit(ctx->stat()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(WHILELOOP expr stat)
std::any AstBuildingVisitor::visitWhileLoop(GazpreaParser::WhileLoopContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::WHILE_LOOP);
    t->addChild(visit(ctx->expr()));
    t->addChild(visit(ctx->stat()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(DOWHILELOOP stat expr)
std::any AstBuildingVisitor::visitDoWhileLoop(GazpreaParser::DoWhileLoopContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::DO_WHILE_LOOP);
    t->addChild(visit(ctx->stat()));
    t->addChild(visit(ctx->expr()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitFuncDecStatement(GazpreaParser::FuncDecStatementContext *ctx)
{
    return visit(ctx->functionDeclaration());
}

std::any AstBuildingVisitor::visitFuncDeclParameter(GazpreaParser::FuncDeclParameterContext *ctx)
{
    return visit(ctx->allTypes());
}

// ^(FUNC_DECL RETURNTYPE NAME FUNC_DECL_PARAMETER_LIST )
std::any AstBuildingVisitor::visitFunctionDeclaration(GazpreaParser::FunctionDeclarationContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::FUNC_DECL);
    t->addChild(visit(ctx->allTypes()));
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol()));
    std::shared_ptr<AST> child3 = std::make_shared<AST>(GazpreaParser::FUNC_DECL_PARAMETER_LIST);
    t->addChild(child3);
    for( auto arg : ctx->funcDeclParameter() ){
        child3->addChild(visit(arg));
    }
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitFuncDefParameter(GazpreaParser::FuncDefParameterContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::FUNC_DEF_PARAM);
    t->addChild(visit(ctx->allTypes()));
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(FUNC_DEF_EXPR_RETURN RETURNTYPE NAME FUNC_DEF_PARAMETER_LIST EXPR)
std::any AstBuildingVisitor::visitExprReturnFunction(GazpreaParser::ExprReturnFunctionContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::FUNC_DEF_EXPR_RETURN);
    t->addChild(visit(ctx->allTypes()));
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol()));
    std::shared_ptr<AST> child3 = std::make_shared<AST>(GazpreaParser::FUNC_DEF_PARAMETER_LIST);
    t->addChild(child3);
    for( auto arg : ctx->funcDefParameter() ){
        child3->addChild(visit(arg));
    }
    t->addChild(visit(ctx->expr()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(FUNC_DEF_BLOCK_RETURN RETURNTYPE NAME FUNC_DEF_PARAMETER_LIST BLOCK)
std::any AstBuildingVisitor::visitBlockEndFunction(GazpreaParser::BlockEndFunctionContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::FUNC_DEF_BLOCK_RETURN);
    t->addChild(visit(ctx->allTypes()));
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol()));
    std::shared_ptr<AST> child3 = std::make_shared<AST>(GazpreaParser::FUNC_DEF_PARAMETER_LIST);
    t->addChild(child3);
    for( auto arg : ctx->funcDefParameter() ){
        child3->addChild(visit(arg));
    }
    t->addChild(visit(ctx->blockStat()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitProcDeclParameter(GazpreaParser::ProcDeclParameterContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::PROC_DECL_PARAM);
    if( ctx->qualifier() != nullptr ){
        t->addChild(visit(ctx->qualifier()));
    }
    else{
        t->addChild(std::make_shared<AST>());
    }
    t->addChild(visit(ctx->allTypes()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitProcDefParameter(GazpreaParser::ProcDefParameterContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::PROC_DEF_PARAM);
    if( ctx->qualifier() != nullptr ){
        t->addChild(visit(ctx->qualifier()));
    }
    else{
        t->addChild(std::make_shared<AST>());
    }
    t->addChild(visit(ctx->allTypes()));
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

// ^(PROC_DECL RETURNTYPE NAME PROC_DECL_PARAMETER_LIST )
std::any AstBuildingVisitor::visitProcedureDeclaration(GazpreaParser::ProcedureDeclarationContext *ctx){
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::PROC_DECL);
    if( ctx->allTypes() != nullptr ){
        t->addChild(visit(ctx->allTypes()));
    }
    else{
        t->addChild(std::make_shared<AST>());
    }
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol()));
    std::shared_ptr<AST> child3 = std::make_shared<AST>(GazpreaParser::PROC_DECL_PARAMETER_LIST);
    t->addChild(visit(child3));
    for( auto arg : ctx->procDeclParameter() ){
        child3->addChild(visit(arg));
    }
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitProcedureDefinition(GazpreaParser::ProcedureDefinitionContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::PROC_DEF);
    if( ctx->allTypes() != nullptr ){
        t->addChild(visit(ctx->allTypes()));
    }
    else{
        t->addChild(std::make_shared<AST>());
    }
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol()));
    std::shared_ptr<AST> child3 = std::make_shared<AST>(GazpreaParser::PROC_DEF_PARAMETER_LIST);
    t->addChild(child3);
    for( auto arg : ctx->procDefParameter() ){
        child3->addChild(visit(arg));
    }
    t->addChild(visit(ctx->blockStat()));
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitProcedureCallStatement(GazpreaParser::ProcedureCallStatementContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::CALLSTAT);
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol()));
    std::shared_ptr<AST> child2 = std::make_shared<AST>(GazpreaParser::ARG_LIST);
    for ( exp : ctx->expr() ){
        child2->addChild(visit(exp));
    }
    t->addChild(child2);
    return std::make_any<std::shared_ptr<AST>>(t);
}

std::any AstBuildingVisitor::visitFuncProcCallExpr(GazpreaParser::FuncProcCallExprContext *ctx)
{
    std::shared_ptr<AST> t = std::make_shared<AST>(GazpreaParser::CALL);
    t->addChild(std::make_shared<AST>(ctx->ID()->getSymbol()));
    std::shared_ptr<AST> child2 = std::make_shared<AST>(GazpreaParser::ARG_LIST);
    for ( exp : ctx->expr() ){
        child2->addChild(visit(exp));
    }
    t->addChild(child2);
    return std::make_any<std::shared_ptr<AST>>(t);
}
