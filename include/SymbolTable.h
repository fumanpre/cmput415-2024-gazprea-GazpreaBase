#ifndef SYMBOLTABLE_H
#define SYMBOLTABLE_H

#include <map>
#include <string>
#include <memory>

#include "Scoping.h"
#include "Symbols.h"

class SymbolTable {
protected:
    void initTypeSystem();
public:	
    std::shared_ptr<GlobalScope> globals;
    SymbolTable();

    std::string toString();
};
#endif