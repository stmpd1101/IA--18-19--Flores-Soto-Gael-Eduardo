class Nodo:
    def __init__(self, valor):
        self.valor = valor
        self.izquierda = None
        self.derecha = None

class Arbol:  # <- nombre pedido en la consigna
    def __init__(self): 
        self.raiz = None

    # ---- Insertar ----
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
        # si valor == nodo_actual.valor, no inserta (sin duplicados)

    # ---- vacio(): boolean ----
    def vacio(self):
        return self.raiz is None

    # ---- buscarNodo(nombre): Nodo ----
    # Usamos búsqueda binaria iterativa por valor (nombre)
    def buscarNodo(self, nombre):
        actual = self.raiz
        while actual is not None:
            if nombre < actual.valor:
                actual = actual.izquierda
            elif nombre > actual.valor:
                actual = actual.derecha
            else:
                return actual
        return None

    # ---- ImprimirArbol ----
    # Opción A: imprimir inorden (ordenado)
    def ImprimirArbol(self):
        print("Árbol (inorden):", self.recorrido_inorden())

    # También dejo tus recorridos por si los necesitas
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
    arbol = Arbol()
    valores = [50, 30, 70, 20, 40, 60, 80]
    for v in valores:
        arbol.insertar(v)

    print("¿Árbol vacío?:", arbol.vacio())
    encontrado = arbol.buscarNodo(60)
    print("buscarNodo(60):", "Encontrado" if encontrado else "No encontrado")

    arbol.ImprimirArbol()
    print("Preorden:", arbol.recorrido_preorden())
    print("Postorden:", arbol.recorrido_postorden())
