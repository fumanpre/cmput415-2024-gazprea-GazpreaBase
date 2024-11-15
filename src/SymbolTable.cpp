#include "SymbolTable.h"

void SymbolTable::initTypeSystem() {
    globals->defineType(std::make_shared<BuiltInTypeSymbol>("integer"));
    globals->defineType(std::make_shared<BuiltInTypeSymbol>("real"));
    globals->defineType(std::make_shared<BuiltInTypeSymbol>("character"));
    globals->defineType(std::make_shared<BuiltInTypeSymbol>("boolean"));
}

SymbolTable::SymbolTable() : globals(std::make_shared<GlobalScope>()) { 
    initTypeSystem(); 
}

std::string SymbolTable::toString() {
    return globals->toString();
}