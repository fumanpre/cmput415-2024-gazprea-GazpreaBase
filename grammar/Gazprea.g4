grammar Gazprea;

tokens{
    FILE,
    VAR_DECL,
    TUPLE_TYPE,
    TUPLE_FIELD,
    DOT_REAL,
    SCIENTIFIC_REAL
}

file:   stat* EOF;

stat:   assignment ';'                              #AssignmentStatement
    |   variableDeclaration ';'                     #DeclarationStatement
    |   'typedef' ( type | sizedVecType | unSizedVecType| sizedMatType | unSizedMatType | tupleType ) ID ';' #TypedefStatement
    |   expr '->' OUTPUTSTREAM ';'                  #OutputStatement
    |   inputStat ';'                               #InputStatement
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
    ;

blockStat:  '{' stat* '}';

functionDeclaration: 'function' ID '(' ( allTypes ID? (',' allTypes ID? )* )? ')' 'returns' allTypes;

functionDefinition: 'function' ID '(' ( allTypes ID (',' allTypes ID )* )? ')' 'returns' allTypes '=' expr ';' #ExprReturnFunction
                |   'function' ID '(' ( allTypes ID (',' allTypes ID )* )? ')' 'returns' allTypes blockStat    #BlockEndFunction
                ;

procedureDeclaration: 'procedure' ID '(' ( qualifier? allTypes ID? (',' qualifier? allTypes ID?)* )? ')' ( 'returns' allTypes )? ';' ;

procedureDefinition: 'procedure' ID '(' ( qualifier? allTypes ID (',' qualifier? allTypes ID)* )? ')' ( blockStat | 'returns' allTypes blockStat );


assignment:     ID '=' expr     #IdAssign
            |   ID '.' (INT | ID) '=' expr    #TupleFieldAssign
            |   ID (',' ID)+ '=' expr         #TupleUnpackAssign
            ;

inputStat:  ID '<-' INPUTSTREAM     #IdInput
        |   ID '.' (INT | ID) '<-' INPUTSTREAM #TupleFieldInput
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


expr:   '(' expr ')'                                                                                                    #ParenthesisExpr
    |   ID '.' (INT | ID)                                                                                               #DotAccessExpr
    |   <assoc=right> op=('+' | '-' | 'not') expr                                                                       #UnaryExpr
    |   <assoc=right> expr '^' expr                                                                                     #ExponentExpr
    |   expr op=('*' | '/' | '%') expr                                                                                  #MultDivRemExpr
    |   expr op=('+' | '-') expr                                                                                        #AddSubExpr
    |   expr op=('<' | '>' | '<=' | '>=') expr                                                                          #LessGreatExpr
    |   expr op=('==' | '!=') expr                                                                                      #EqNotEqExpr
    |   expr 'and' expr                                                                                                 #BooleanAndExpr
    |   expr op=('or'|'xor')                                                                                            #BooleanOrExpr
    |   'as' '<' ( type | sizedVecType | unSizedVecType| sizedMatType | unSizedMatType | tupleType ) '>' '(' expr ')'   #TypeCastExpr
    |   '(' expr ( ',' expr )+ ')'                                                                                      #TupleLiteralExpr
    |   STRING                                                                                                          #StringLiteralExpr
    |   real                                                                                                            #RealLiteralExpr
    |   BOOL                                                                                                            #BoolLiteralExpr
    |   CHAR                                                                                                            #CharLiteralExpr
    |   INT                                                                                                             #IntLiteralExpr
    |   ID                                                                                                              #IdExpr
    ;

real:   INT ('e'|'E') INT                   #IntEReal
    |   real ('e'|'E') INT                  #RealEReal  
    |   INT '.' INT?                        #PostDotReal
    |   INT? '.' INT                        #PreDotReal
    ;

// Lexer Rules
OUTPUTSTREAM:   'std_output';
INPUTSTREAM:    'std_input';
XOR         : 'xor';
AND         : 'and';
AS          : 'as';
BOOLEAN     : 'boolean';
BREAK       : 'break';
BY          : 'by';
CALL        : 'call';
CHARACTER   : 'character';
COLUMNS     : 'columns';
CONST       : 'const';
CONTINUE    : 'continue';
ELSE        : 'else';
FORMAT      : 'format';
FUNCTION    : 'function';
IF          : 'if';
IN          : 'in';
INTEGER     : 'integer';
LENGTH      : 'length';
LOOP        : 'loop';
NOT         : 'not';
OR          : 'or';
PROCEDURE   : 'procedure';
REAL        : 'real';
RETURN      : 'return';
RETURNS     : 'returns';
REVERSE     : 'reverse';
ROWS        : 'rows';
TUPLE       : 'tuple';
TYPEDEF     : 'typedef';
VAR         : 'var';
WHILE       : 'while';



// Literals
STRING: '"' ( ESCAPE | . )*? '"';
CHAR:   '\''  ( ESCAPE | . ) '\'';
BOOL:   ('true' | 'false');
ID:     ( '_' | ALPHA) (ALPHA | DIGIT | '_')* ;
INT:	( '+' | '-')? DIGIT+;

// OPERATORS USED IN AST
DOT: '.';


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