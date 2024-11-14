#ifndef ASTBUILDINGVISITOR_H
#define ASTBUILDINGVISITOR_H
#include "GazpreaBaseVisitor.h"

using namespace gazprea;

class AstBuildingVisitor : public GazpreaBaseVisitor {
    public:

    std::any visitFile(GazpreaParser::FileContext *ctx) override;

    std::any visitDeclarationStatement(GazpreaParser::DeclarationStatementContext *ctx);

    std::any visitDecl(GazpreaParser::DeclContext *ctx) override;

    std::any visitInferredDecl(GazpreaParser::InferredDeclContext *ctx) override;

    std::any visitVarSizedDecl(GazpreaParser::VarSizedDeclContext *ctx) override;

    std::any visitAssignmentStatement(GazpreaParser::AssignmentStatementContext *ctx);

    std::any visitQualifier(GazpreaParser::QualifierContext *ctx) override;

    std::any visitType(GazpreaParser::TypeContext *ctx) override;

    std::any visitTupleType(GazpreaParser::TupleTypeContext *ctx) override;

    std::any visitTupleField(GazpreaParser::TupleFieldContext *ctx) override;

    std::any visitIdLVal(GazpreaParser::IdLValContext *ctx) override;

    std::any visitTupleAccessLVal(GazpreaParser::TupleAccessLValContext *ctx) override;

    std::any visitLValAssign(GazpreaParser::LValAssignContext *ctx) override;

    std::any visitMultiAssign(GazpreaParser::MultiAssignContext *ctx) override;

    std::any visitParenthesisExpr(GazpreaParser::ParenthesisExprContext *ctx) override;

    std::any visitIdExpr(GazpreaParser::IdExprContext *ctx) override;

    std::any visitIntLiteralExpr(GazpreaParser::IntLiteralExprContext *ctx) override;

    std::any visitBoolLiteralExpr(GazpreaParser::BoolLiteralExprContext *ctx) override;

    std::any visitCharLiteralExpr(GazpreaParser::CharLiteralExprContext *ctx) override;

    std::any visitPreDotReal(GazpreaParser::PreDotRealContext *ctx) override;

    std::any visitPostDotReal(GazpreaParser::PostDotRealContext *ctx) override;

    std::any visitIntEReal(GazpreaParser::IntERealContext *ctx) override;

    std::any visitRealEReal(GazpreaParser::RealERealContext *ctx) override;

    std::any visitDotAccessExpr(GazpreaParser::DotAccessExprContext *ctx) override;

    std::any visitUnaryExpr(GazpreaParser::UnaryExprContext *ctx) override;

    std::any visitExponentExpr(GazpreaParser::ExponentExprContext *ctx) override;

    std::any visitMultDivRemExpr(GazpreaParser::MultDivRemExprContext *ctx) override;

    std::any visitAddSubExpr(GazpreaParser::AddSubExprContext *ctx) override;

    std::any visitEqNotEqExpr(GazpreaParser::EqNotEqExprContext *ctx) override;

    std::any visitLessGreatExpr(GazpreaParser::LessGreatExprContext *ctx) override;

    std::any visitBooleanAndExpr(GazpreaParser::BooleanAndExprContext *ctx) override;

    std::any visitBooleanOrExpr(GazpreaParser::BooleanOrExprContext *ctx) override;

    //std::any visitTypeCastExpr(GazpreaParser::TypeCastExprContext *ctx) override;

    //std::any visitTupleLiteralExpr(GazpreaParser::TupleLiteralExprContext *ctx) override;


    /*

    std::any visitTypedefStatement(GazpreaParser::TypedefStatementContext *ctx) override;

    std::any visitOutputStatement(GazpreaParser::OutputStatementContext *ctx) override;

    std::any visitIfStatement(GazpreaParser::IfStatementContext *ctx) override;

    std::any visitBreakStatement(GazpreaParser::BreakStatementContext *ctx) override;

    std::any visitContinueStatement(GazpreaParser::ContinueStatementContext *ctx) override;

    std::any visitReturnStatement(GazpreaParser::ReturnStatementContext *ctx) override;

    std::any visitBlockStat(GazpreaParser::BlockStatContext *ctx) override;

    std::any visitFunctionDeclaration(GazpreaParser::FunctionDeclarationContext *ctx) override;

    std::any visitExprReturnFunction(GazpreaParser::ExprReturnFunctionContext *ctx) override;

    std::any visitBlockEndFunction(GazpreaParser::BlockEndFunctionContext *ctx) override;

    std::any visitProcedureDeclaration(GazpreaParser::ProcedureDeclarationContext *ctx) override;

    std::any visitProcedureDefinition(GazpreaParser::ProcedureDefinitionContext *ctx) override;

    std::any visitLValInput(GazpreaParser::LValInputContext *ctx) override;

    std::any visitInfiniteLoop(GazpreaParser::InfiniteLoopContext *ctx) override;

    std::any visitWhileLoop(GazpreaParser::WhileLoopContext *ctx) override;

    std::any visitDoWhileLoop(GazpreaParser::DoWhileLoopContext *ctx) override;

    std::any visitSizedVecType(GazpreaParser::SizedVecTypeContext *ctx) override;

    std::any visitUnSizedVecType(GazpreaParser::UnSizedVecTypeContext *ctx) override;

    std::any visitSizedMatType(GazpreaParser::SizedMatTypeContext *ctx) override;

    std::any visitUnSizedMatType(GazpreaParser::UnSizedMatTypeContext *ctx) override;

    std::any visitStringLiteralExpr(GazpreaParser::StringLiteralExprContext *ctx) override;

    */

    
};
#endif