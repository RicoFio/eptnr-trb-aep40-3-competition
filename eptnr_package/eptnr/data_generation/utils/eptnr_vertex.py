from ...constants.igraph_colors import IGraphColors
from ...constants.igraph_vertex_types import IGraphVertexTypes
from dataclasses import dataclass, field


@dataclass
class EPTNRVertex:
    x: float
    y: float
    color: IGraphColors
    type: IGraphVertexTypes
    vertex_index: int = field(init=False, repr=False)

    @property
    def name(self) -> str:
        if self.vertex_index is None:
            raise AttributeError("Vertex index has not been set yet")
        return self.type.name.strip('_NODE') + str(self.vertex_index)
