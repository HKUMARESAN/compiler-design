PROGRAM:(3)
//art_expr.l
%{
#include<stdio.h>
#include "y.tab.h"
%}
%%
[a-zA-Z][0-9a-zA-Z]* {return ID;}
[0-9]+ {return DIG;}
[ \t]+ {;}
. {return yytext[0];}
\n {return 0;}
%%
int yywrap()
{
return 1;
}
//art_expr.y
%{
#include<stdio.h>
%}
%token ID DIG
%left '+''-'
%left '*''/'
%right UMINUS
%%
stmt:expn ;
expn:expn'+'expn
|expn'-'expn
|expn'*'expn
|expn'/'expn
|'-'expn %prec UMINUS
|'('expn')'
|DIG
|ID
;
%%
int main()
{
printf("Enter the Expression \n");
yyparse();
printf("valid Expression \n");
return 0;
}
int yyerror()
{
printf("Invalid Expression");
exit(0);
}