class Nodo:
    def __init__(self, valor):
        self.valor = valor
        self.izquierda = None
        self.derecha = None

class ArbolBinarioBusqueda:
    def __init__(self): 
        self.raiz = None

    def insertar(self, valor):
        if self.raiz is None:
            self.raiz = Nodo(valor)
        else:
            self._insertar_recursivo(self.raiz, valor)

    def _insertar_recursivo(self, nodo_actual, valor):
        if valor < nodo_actual.valor:
            if nodo_actual.izquierda is None:
                nodo_actual.izquierda = Nodo(valor)
            else:
                self._insertar_recursivo(nodo_actual.izquierda, valor)
        elif valor > nodo_actual.valor:
            if nodo_actual.derecha is None:
                nodo_actual.derecha = Nodo(valor)
            else:
                self._insertar_recursivo(nodo_actual.derecha, valor)

    def recorrido_inorden(self):
        return self._inorden(self.raiz)

    def _inorden(self, nodo):
        if nodo is None:
            return []
        return self._inorden(nodo.izquierda) + [nodo.valor] + self._inorden(nodo.derecha)

    def recorrido_preorden(self):
        return self._preorden(self.raiz)

    def _preorden(self, nodo):
        if nodo is None:
            return []
        return [nodo.valor] + self._preorden(nodo.izquierda) + self._preorden(nodo.derecha)

    def recorrido_postorden(self):
        return self._postorden(self.raiz)

    def _postorden(self, nodo):
        if nodo is None:
            return []
        return self._postorden(nodo.izquierda) + self._postorden(nodo.derecha) + [nodo.valor]

if __name__ == "__main__":
    arbol = ArbolBinarioBusqueda()
    valores = [50, 30, 70, 20, 40, 60, 80]
    for v in valores:
        arbol.insertar(v)

    print("Recorrido Inorden:", arbol.recorrido_inorden())
    print("Recorrido Preorden:", arbol.recorrido_preorden())
    print("Recorrido Postorden:", arbol.recorrido_postorden())