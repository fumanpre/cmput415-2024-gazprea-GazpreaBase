grammar Gazprea;

// may combine fnc and proc nodes for part 2 as they should be mostly similar
tokens{
    FILE,
    VAR_DECL,
    TUPLE_TYPE,
    TUPLE_FIELD,
    DOT_REAL,
    SCIENTIFIC_REAL,
    TUPLE_ACCESS,
    MULTI_ASSIGN,
    UNARY_MINUS,
    TUPLE_LITERAL,
    BLOCK,
    INF_LOOP,
    WHILE_LOOP,
    DO_WHILE_LOOP,
    FUNC_DECL,
    FUNC_DEF_EXPR_RETURN,
    FUNC_DEF_BLOCK_RETURN,
    FUNC_DEF_PARAM,
    FUNC_DECL_PARAMETER_LIST,
    FUNC_DEF_PARAMETER_LIST,
    PROC_DECL_PARAM,
    PROC_DEF_PARAM,
    PROC_DECL,
    PROC_DEF,
    PROC_DECL_PARAMETER_LIST,
    PROC_DEF_PARAMETER_LIST,
    ARG_LIST,
    CALL
}

file:   stat* EOF;

stat:   assignment ';'                              #AssignmentStatement
    |   variableDeclaration ';'                     #DeclarationStatement
    |   'typedef' allTypes ID ';'                   #TypedefStatement
    |   expr '->' OUTPUTSTREAM ';'                  #OutputStatement
    |   lVal '<-' INPUTSTREAM ';'                   #InputStatement
    |   blockStat                                   #BlockStatement
    |   'if' '(' expr ')' stat ('else' stat)?       #IfStatement
    |   loopStat                                    #LoopStatement
    |   'break' ';'                                 #BreakStatement
    |   'continue' ';'                              #ContinueStatement
    |   'return' expr? ';'                          #ReturnStatement
    |   functionDeclaration ';'                     #FuncDecStatement
    |   functionDefinition                          #FuncDefStatement
    |   procedureDeclaration                        #ProcDecStatement
    |   procedureDefinition                         #ProcDefStatement
    |   'call' ID '(' ( expr (',' expr )* )? ')' ';'#ProcedureCallStatement
    ;

blockStat:  '{' stat* '}';

funcDeclParameter : allTypes ID?;
functionDeclaration: 'function' ID '(' ( funcDeclParameter (',' funcDeclPrameter )* )? ')' 'returns' allTypes;

funcDefParameter: allTypes ID;
functionDefinition: 'function' ID '(' ( funcDefParameter (',' funcDefParameter )* )? ')' 'returns' allTypes '=' expr ';' #ExprReturnFunction
                |   'function' ID '(' ( funcDefParameter (',' funcDefParameter )* )? ')' 'returns' allTypes blockStat    #BlockEndFunction
                ;

procDeclParameter : qualifier? allTypes ID?;
procedureDeclaration: 'procedure' ID '(' ( procDeclParameter (',' procDeclParameter )* )? ')' ( 'returns' allTypes )? ';' ;

procDefParameter: qualifier? allTypes ID;
procedureDefinition: 'procedure' ID '(' ( procDefParameter (',' procDefParameter )* )? ')' ( 'returns' allTypes )? blockStat ;

lVal:   ID                  #IdLVal
    |   ID '.' (INT | ID)   #TupleAccessLVal
    ;

assignment:     lVal '=' expr     #LValAssign
            |   lVal (',' lVal)+ '=' expr         #MultiAssign
            ;

//TODO iterator loops
loopStat:   'loop' stat     #InfiniteLoop
        |   'loop' 'while' '(' expr ')' stat        #WhileLoop
        |   'loop' stat 'while' '(' expr ')' ';'    #DoWhileLoop
        ;

variableDeclaration:    qualifier? fixedSizeType ID ('=' expr)?   #Decl
                    |   qualifier ID '=' expr           #InferredDecl
                    |   qualifier? varSizedType ID '=' expr     #VarSizedDecl
                    ;

qualifier: 'var'
         | 'const'
         ;

type: 'boolean'
    | 'character'
    | 'integer'
    | 'real'
    | 'string'
    | ID // this is for typedef types
    ;

tupleFieldType: ( type | sizedVecType | sizedMatType );
fixedSizeType: ( type | sizedVecType | sizedMatType | tupleType );
varSizedType: ( unSizedVecType | unSizedMatType );
allTypes:   ( type | sizedVecType | unSizedVecType| sizedMatType | unSizedMatType | tupleType );

sizedVecType:   type '[' INT ']';
unSizedVecType: type '[' '*' ']';
sizedMatType:   type '[' INT ',' INT ']' ;
unSizedMatType: type '[' ( INT | '*' ) ',' ( INT | '*' ) ']';
tupleType:  'tuple' '(' tupleField ( ',' tupleField )+ ')' ;

tupleField:  tupleFieldType ID?;


expr:   ID '(' ( expr (',' expr )* )? ')'                                                                               #FuncProcCallExpr
    |   '(' expr ')'                                                                                                    #ParenthesisExpr
    |   ID '.' (INT | ID)                                                                                               #DotAccessExpr
    |   <assoc=right> op=('+' | '-' | BOOLEAN_NOT ) expr                                                                #UnaryExpr
    |   <assoc=right> expr '^' expr                                                                                     #ExponentExpr
    |   expr op=('*' | '/' | '%') expr                                                                                  #MultDivRemExpr
    |   expr op=('+' | '-') expr                                                                                        #AddSubExpr
    |   expr op=('<' | '>' | '<=' | '>=') expr                                                                          #LessGreatExpr
    |   expr op=('==' | '!=') expr                                                                                      #EqNotEqExpr
    |   expr 'and' expr                                                                                                 #BooleanAndExpr
    |   expr op=('or'|'xor')  expr                                                                                      #BooleanOrExpr
    |   'as' '<' allTypes '>' '(' expr ')'                                                                              #TypeCastExpr
    |   '(' expr ( ',' expr )+ ')'                                                                                      #TupleLiteralExpr
    |   STRING                                                                                                          #StringLiteralExpr
    |   real                                                                                                            #RealLiteralExpr
    |   BOOL                                                                                                            #BoolLiteralExpr
    |   CHAR                                                                                                            #CharLiteralExpr
    |   INT                                                                                                             #IntLiteralExpr
    |   ID                                                                                                              #IdExpr
    ;

// THIS WOULD FAIL SOME TEST CASES LIKE REAL declaration 35e4 and also won't let e or E be an ID. SHOULD FIX BEFORE PART 2
real:   INT ('e'|'E') INT                   #IntEReal
    |   real ('e'|'E') INT                  #RealEReal
    |   INT '.' INT?                        #PostDotReal
    |   INT? '.' INT                        #PreDotReal
    ;

// Lexer Rules
OUTPUTSTREAM:   'std_output';
INPUTSTREAM:    'std_input';
TYPE_CAST_EXPR: 'as';
TYPEDEF     : 'typedef';
IF          : 'if';
ELSE        : 'else';
BREAK       : 'break';
CONTINUE    : 'continue';
CALLSTAT        : 'call';

BOOLEAN     : 'boolean';
BY          : 'by';
CHARACTER   : 'character';
COLUMNS     : 'columns';
CONST       : 'const';
FORMAT      : 'format';
FUNCTION    : 'function';
IN          : 'in';
INTEGER     : 'integer';
LENGTH      : 'length';
LOOP        : 'loop';
PROCEDURE   : 'procedure';
REAL        : 'real';
RETURN      : 'return';
RETURNS     : 'returns';
REVERSE     : 'reverse';
ROWS        : 'rows';
TUPLE       : 'tuple';
VAR         : 'var';
WHILE       : 'while';



// Literals
STRING: '"' ( ESCAPE | . )*? '"';
CHAR:   '\''  ( ESCAPE | . ) '\'';
BOOL:   ('true' | 'false');
INT:	( '+' | '-')? DIGIT+;

// OPERATORS
DOT: '.';
ASSIGN: '=';
BOOLEAN_NOT         : 'not';
EXPONENT:   '^';
MULT:   '*';
DIV:    '/';
REM:    '%';
ADD:    '+';
SUB:    '-';
EQUALS: '==';
NOTEQUALS:  '!=';
LESS:   '<';
GREATER:  '>';
LESSEQUAL:  '<=';
GREATEREQUAL:   '>=';
BOOLEAN_XOR         : 'xor';
BOOLEAN_AND         : 'and';
BOOLEAN_OR         : 'or';


ID:     ( '_' | ALPHA) (ALPHA | DIGIT | '_')* ;

// Skip whitespace
WS : [ \t\r\n]+ -> skip ;
fragment
ALPHA	:	[a-zA-Z];
fragment
DIGIT	:	[0-9];
fragment
ESCAPE: '\\' ('0' | 'a' | 'b' | 't' | 'n' | 'r' | '"' | '\'' | '\\');

// Comments
COMMENT:	'//' .*? ( '\n' | EOF ) -> skip;

COMMENT_BLOCK:  '/*' .*? '*/' -> skip;