#include "GazpreaLexer.h"
#include "GazpreaParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"
#include "tree/ParseTreeWalker.h"

#include "BackEnd.h"
#include "Ast.h"
#include "AstBuildingVisitor.h"

#include <iostream>
#include <fstream>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Missing required argument.\n"
              << "Required arguments: <input file path> <output file path>\n";
    return 1;
  }

  // Open the file then parse and lex it.
  antlr4::ANTLRFileStream afs;
  afs.loadFromFile(argv[1]);
  gazprea::GazpreaLexer lexer(&afs);
  antlr4::CommonTokenStream tokens(&lexer);
  gazprea::GazpreaParser parser(&tokens);

  // Get the root of the parse tree. Use your base rule name.
  antlr4::tree::ParseTree *tree = parser.file();

  // HOW TO USE A LISTENER
  // Make the listener
  // MyListener listener;
  // Walk the tree
  // antlr4::tree::ParseTreeWalker::DEFAULT.walk(&listener, tree);

  // HOW TO USE A VISITOR
  // Make the visitor
  AstBuildingVisitor visitor;
  // Visit the tree
  std::shared_ptr<AST> a = std::any_cast<std::shared_ptr<AST>>(visitor.visit(tree));
  std::cout<<(a->toStringTree()) << "\n";
  std::cout<<(a->toStringTree2()) << "\n";

  // BackEnd backend;
  // backend.emitMain();

  // HOW TO WRITE OUT.
  // std::ofstream out(argv[2]);
  // out << "This is out...\n";

  return 0;
}
