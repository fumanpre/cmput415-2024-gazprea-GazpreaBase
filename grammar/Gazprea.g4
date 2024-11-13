grammar Gazprea;

file:   stat* EOF;

stat:   assignment ';'                              #assignmentStat
    |   variableDeclaration ';'                     #declarationStat
    |   'typedef' ( type | sizedVecType | unSizedVecType| sizedMatType | unSizedMatType | tupleType ) ID ';' #typedefStat
    |   expr '->' OUTPUTSTREAM ';'                  #outputStat
    |   inputStat ';'                               #inputStat
    |   blockStat                                   #blockStat
    |   'if' '(' expr ')' stat ('else' stat)?       #ifStat
    |   loopStat                                    #loopStat
    |   'break' ';'                                 #breakStat
    |   'continue' ';'                              #continueStat
    |   'return' expr? ';'                          #returnStat
    |   functionDeclaration ';'                     #funcDecStat
    |   functionDefinition                          #funcDefStat
    ;

blockStat:  '{' stat* '}';

functionDeclaration: 'function' ID '(' ( allTypes ID? (',' allTypes ID? )* )? ')' 'returns' allTypes;

functionDefinition: 'function' ID '(' ( allTypes ID (',' allTypes ID )* )? ')' 'returns' allTypes '=' expr ';' #exprReturnFunction
                |   'function' ID '(' ( allTypes ID (',' allTypes ID )* )? ')' 'returns' allTypes blockStat    #blockEndFunction
                ;

assignment:     ID '=' expr     #idAssign
            |   ID '.' (INT | ID) '=' expr    #tupleFieldAssign
            |   ID (, ID)+ '=' expr             #tupleUnpackAssign
            ;

inputStat:  ID '<-' INPUTSTREAM     #idInput
        |   ID '.' (INT | ID) '<-' INPUTSTREAM #tupleFieldInput
        ;

//TODO iterator loops
loopStat:   'loop' stat     #infiniteLoop
        |   'loop' 'while' '(' expr ')' stat        #whileLoop
        |   'loop' stat 'while' '(' expr ')' ';'    #doWhileLoop
        ;

variableDeclaration:   qualifier? type ID ('=' expr)? #decl
                    |   qualifier ID '=' expr #inferredDecl
                    |   vectorDeclaration #vecDecl
                    |   matrixDeclaration #matDecl
                    |   tupleType ID ('=' expr)? #tupleDecl
                    ;

vectorDeclaration:  sizedVecType ID ('=' expr)? #sizedVecDecl
                |   unSizedVecType ID '=' expr #unSizedVecDecl
                ;

matrixDeclaration:  sizedMatType ID ('=' expr)? #sizedMatDecl
                |   unSizedMatType ID '=' expr #unSizedMatDecl
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

allTypes:   ( type | sizedVecType | unSizedVecType| sizedMatType | unSizedMatType | tupleType );

sizedVecType:   type '[' INT ']';
unSizedVecType: type '[' '*' ']';
sizedMatType:   type '[' INT ',' INT ']' ;
unSizedMatType: type '[' ( INT | '*' ) ',' ( INT | '*' ) ']';
tupleType:  'tuple' '(' ( type | sizedVecType | sizedMatType ) ID? ( ',' ( type | sizedVecType | sizedMatType ) ID? )+ ')' ;


expr:   '(' expr ')'                                                                                                    #paranthesisExpr
    |   ID '.' (INT | ID)                                                                                               #dotAccessExpr
    |   <assoc=right> op=('+' | '-' | 'not') expr                                                                       #unaryExpr
    |   <assoc=right> expr '^' expr                                                                                     #exponentExpr
    |   expr op=('*' | '/' | '%') expr                                                                                  #multDivRemExpr
    |   expr op=('+' | '-') expr                                                                                        #addSubExpr
    |   expr op=('<' | '>' | '<=' | '>=') expr                                                                          #ltgtComparisonExpr
    |   expr op=('==' | '!=') expr                                                                                      #eqComparisonExpr
    |   expr 'and' expr                                                                                                 #booleanAndExpr
    |   expr op=('or'|'xor')                                                                                            #booleanOrExpr
    |   'as' '<' ( type | sizedVecType | unSizedVecType| sizedMatType | unSizedMatType | tupleType ) '>' '(' expr ')'   #typeCastExpr
    |   '(' expr ( ',' expr )+ ')'                                                                                      #tupleLiteralExpr
    |   STRING                                                                                                          #stringLiteralExpr
    |   real                                                                                                            #realLiteralExpr    
    |   BOOL                                                                                                            #boolLiteralExpr
    |   CHAR                                                                                                            #charLiteralExpr
    |   INT                                                                                                             #intLiteralExpr
    |   ID                                                                                                              #idExpr
    ;

real:   INT '.' INT?
    |   INT? '.' INT
    |   ( real | INT ) ('e'|'E') INT
    ;

// Lexer Rules
OUTPUTSTREAM:   'std_output';
INPUTSTREAM:    'std_input';

// Literals
STRING: '"' ( ESCAPE | . )*? '"';
CHAR:   '\''  ( ESCAPE | . ) '\'';
BOOL:   ('true' | 'false');
ID:     ( '_' | ALPHA) (ALPHA | DIGIT | '_')* ;
INT:	'-'? DIGIT+;




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