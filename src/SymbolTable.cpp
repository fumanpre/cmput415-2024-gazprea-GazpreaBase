#include "SymbolTable.h"

void SymbolTable::initTypeSystem() {
    globals->defineType(std::make_shared<BuiltInTypeSymbol>("integer", INT));
    globals->defineType(std::make_shared<BuiltInTypeSymbol>("real", REAL));
    globals->defineType(std::make_shared<BuiltInTypeSymbol>("character", CHAR));
    globals->defineType(std::make_shared<BuiltInTypeSymbol>("boolean", BOOL));
}

SymbolTable::SymbolTable() : globals(std::make_shared<GlobalScope>()) { 
    initTypeSystem(); 
}

std::string SymbolTable::toString() {
    return globals->toString();
}