from abc import ABC, abstractmethod
import base64
import pickle
from typing import Set, Optional, Union, Iterable
try:
    from bitarray import bitarray
    BITARRAY_AVAILABLE = True
except ImportError:
    BITARRAY_AVAILABLE = False


class DataSketch(ABC):
    """
    Abstract base class for data sketches used in decision tree construction.
    
    A DataSketch represents a set of row IDs from the training data and provides
    efficient set operations (intersection, union, subtraction) and serialization capabilities.
    Used to track which data points satisfy specific feature conditions or 
    target class values.
    """
    
    @abstractmethod
    def intersect(self, other: 'DataSketch') -> 'DataSketch':
        """
        Returns a new DataSketch containing the intersection of row IDs
        between this sketch and another sketch.
        
        Args:
            other (DataSketch): Another DataSketch to intersect with
            
        Returns:
            DataSketch: New sketch containing only row IDs present in both sketches
            
        Raises:
            TypeError: If other is not a DataSketch of the same type
        """
        pass
    
    @abstractmethod
    def union(self, other: 'DataSketch') -> 'DataSketch':
        """
        Returns a new DataSketch containing the union of row IDs
        between this sketch and another sketch.
        
        Args:
            other (DataSketch): Another DataSketch to union with
            
        Returns:
            DataSketch: New sketch containing row IDs present in either sketch
            
        Raises:
            TypeError: If other is not a DataSketch of the same type
        """
        pass
    
    @abstractmethod
    def subtract(self, other: 'DataSketch') -> 'DataSketch':
        """
        Returns a new DataSketch containing row IDs that are in this sketch
        but not in the other sketch (set difference: self - other).
        
        Args:
            other (DataSketch): DataSketch to subtract from this one
            
        Returns:
            DataSketch: New sketch containing row IDs in self but not in other
            
        Raises:
            TypeError: If other is not a DataSketch of the same type
            NotImplementedError: If sketch type doesn't support subtraction
        """
        pass
    
    @abstractmethod
    def add_row_id(self, row_id: int) -> None:
        """
        Adds a row ID to this DataSketch.
        
        Args:
            row_id (int): Row ID to add to the sketch
            
        Raises:
            ValueError: If row_id is invalid for this sketch type
        """
        pass
    
    @abstractmethod
    def get_count(self) -> int:
        """
        Returns the number of row IDs contained in this sketch.
        
        Returns:
            int: Count of unique row IDs in the sketch
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_base64(cls, encoded_string: str, **kwargs) -> 'DataSketch':
        """
        Creates a DataSketch instance from a Base64 encoded string.
        
        Args:
            encoded_string (str): Base64 encoded representation of the sketch
            **kwargs: Additional parameters specific to sketch type
            
        Returns:
            DataSketch: New DataSketch instance decoded from the string
            
        Raises:
            ValueError: If the encoded string is invalid or corrupted
        """
        pass
    
    @abstractmethod
    def to_base64(self) -> str:
        """
        Serializes this DataSketch to a Base64 encoded string.
        
        Returns:
            str: Base64 encoded representation of the sketch
        """
        pass
    
    def __len__(self) -> int:
        """
        Returns the count of row IDs in the sketch.
        Convenience method that calls get_count().
        
        Returns:
            int: Count of unique row IDs in the sketch
        """
        return self.get_count()
    
    def is_empty(self) -> bool:
        """
        Checks if the sketch contains any row IDs.
        
        Returns:
            bool: True if sketch is empty, False otherwise
        """
        return self.get_count() == 0


class SketchFactory:
    """
    Factory class for creating DataSketch instances.
    Provides abstraction layer for sketch type selection.
    """
    
    SKETCH_TYPES = {
        'bitvector': None,  # Will be set after BitVector is defined
        'thetasketch': None,  # Will be set when ThetaSketch is implemented
    }
    
    @classmethod
    def create_sketch(cls, sketch_type: str, row_ids: Optional[Union[Set[int], Iterable[int]]] = None, **kwargs) -> DataSketch:
        """
        Create a new sketch of the specified type.
        
        Args:
            sketch_type (str): Type of sketch ('bitvector' or 'thetasketch')
            row_ids (Set[int] or Iterable[int], optional): Initial row IDs
            **kwargs: Additional parameters for sketch creation
            
        Returns:
            DataSketch: New sketch instance
            
        Raises:
            ValueError: If sketch_type is not supported
        """
        sketch_type = sketch_type.lower()
        
        if sketch_type not in cls.SKETCH_TYPES:
            raise ValueError(f"Unsupported sketch type: {sketch_type}. Supported types: {list(cls.SKETCH_TYPES.keys())}")
        
        sketch_class = cls.SKETCH_TYPES[sketch_type]
        if sketch_class is None:
            raise ValueError(f"Sketch type '{sketch_type}' not available")
        
        return sketch_class(row_ids, **kwargs)
    
    @classmethod
    def from_base64(cls, encoded_string: str, sketch_type: str, **kwargs) -> DataSketch:
        """
        Create a sketch from Base64 encoded data.
        
        Args:
            encoded_string (str): Base64 encoded sketch data
            sketch_type (str): Type of sketch to create
            **kwargs: Additional parameters for sketch creation
            
        Returns:
            DataSketch: Decoded sketch instance
        """
        sketch_type = sketch_type.lower()
        
        if sketch_type not in cls.SKETCH_TYPES:
            raise ValueError(f"Unsupported sketch type: {sketch_type}")
        
        sketch_class = cls.SKETCH_TYPES[sketch_type]
        if sketch_class is None:
            raise ValueError(f"Sketch type '{sketch_type}' not available")
        
        return sketch_class.from_base64(encoded_string, **kwargs)
    
    @classmethod
    def register_sketch_type(cls, type_name: str, sketch_class):
        """
        Register a new sketch type.
        
        Args:
            type_name (str): Name of the sketch type
            sketch_class: Class that implements DataSketch interface
        """
        cls.SKETCH_TYPES[type_name.lower()] = sketch_class


class BitVector(DataSketch):
    """
    True BitVector implementation using bitarray for efficient bitwise operations.
    
    Uses actual bit arrays with bitwise AND/OR operations for maximum efficiency.
    Requires knowing the maximum row ID (universe size) upfront.
    
    Falls back to set-based implementation if bitarray package is not available.
    """
    
    def __init__(self, row_ids: Optional[Union[Set[int], Iterable[int]]] = None, 
                 universe_size: Optional[int] = None):
        """
        Initialize BitVector with row IDs and universe size.
        
        Args:
            row_ids (Set[int] or Iterable[int], optional): Row IDs to include
            universe_size (int, optional): Maximum possible row ID + 1.
                                         Required for true bitarray implementation.
                                         If None, falls back to set-based approach.
        """
        self.universe_size = universe_size
        
        if BITARRAY_AVAILABLE and universe_size is not None:
            # True BitVector using bitarray
            self._use_bitarray = True
            self.bits = bitarray(universe_size)
            self.bits.setall(0)  # Initialize all bits to 0
            
            if row_ids:
                for row_id in row_ids:
                    if 0 <= row_id < universe_size:
                        self.bits[row_id] = 1
                    else:
                        raise ValueError(f"Row ID {row_id} outside universe size {universe_size}")
        else:
            # Fallback to set-based implementation
            self._use_bitarray = False
            self.row_ids: Set[int] = set(row_ids) if row_ids else set()
            if not BITARRAY_AVAILABLE and universe_size is not None:
                print("Warning: bitarray package not available. Install with: pip install bitarray")
    
    def intersect(self, other: 'DataSketch') -> 'BitVector':
        """
        Returns intersection using efficient bitwise AND or set intersection.
        
        Args:
            other (DataSketch): Another DataSketch to intersect with
            
        Returns:
            BitVector: New BitVector containing intersection
            
        Raises:
            TypeError: If other is not a BitVector
            ValueError: If universe sizes don't match for bitarray mode
        """
        if not isinstance(other, BitVector):
            raise TypeError(f"Cannot intersect BitVector with {type(other)}")
        
        if self._use_bitarray and other._use_bitarray:
            # True bitwise AND operation - O(n/8) where n is universe_size
            if self.universe_size != other.universe_size:
                raise ValueError("Cannot intersect BitVectors with different universe sizes")
            
            result = BitVector(universe_size=self.universe_size)
            result.bits = self.bits & other.bits  # Bitwise AND
            return result
            
        elif not self._use_bitarray and not other._use_bitarray:
            # Set intersection fallback
            intersected_ids = self.row_ids.intersection(other.row_ids)
            return BitVector(intersected_ids, self.universe_size)
        else:
            raise ValueError("Cannot intersect BitVector implementations of different types")
    
    def union(self, other: 'BitVector') -> 'BitVector':
        """
        Returns union using efficient bitwise OR or set union.
        
        Args:
            other (BitVector): Another BitVector to union with
            
        Returns:
            BitVector: New BitVector containing union
        """
        if not isinstance(other, BitVector):
            raise TypeError(f"Cannot union BitVector with {type(other)}")
        
        if self._use_bitarray and other._use_bitarray:
            if self.universe_size != other.universe_size:
                raise ValueError("Cannot union BitVectors with different universe sizes")
            
            result = BitVector(universe_size=self.universe_size)
            result.bits = self.bits | other.bits  # Bitwise OR
            return result
        elif not self._use_bitarray and not other._use_bitarray:
            union_ids = self.row_ids.union(other.row_ids)
            return BitVector(union_ids, self.universe_size)
        else:
            raise ValueError("Cannot union BitVector implementations of different types")
    
    def subtract(self, other: 'BitVector') -> 'BitVector':
        """
        Returns BitVector with row IDs in this sketch but not in other.
        
        Args:
            other (BitVector): BitVector to subtract
            
        Returns:
            BitVector: New BitVector with set difference
        """
        if not isinstance(other, BitVector):
            raise TypeError(f"Cannot subtract {type(other)} from BitVector")
        
        if self._use_bitarray and other._use_bitarray:
            if self.universe_size != other.universe_size:
                raise ValueError("Cannot subtract BitVectors with different universe sizes")
            
            result = BitVector(universe_size=self.universe_size)
            # Bitwise subtraction: self AND (NOT other)
            result.bits = self.bits & (~other.bits)
            return result
            
        elif not self._use_bitarray and not other._use_bitarray:
            # Set subtraction
            result_ids = self.row_ids - other.row_ids
            return BitVector(result_ids, self.universe_size)
        else:
            raise ValueError("Cannot subtract BitVector implementations of different types")
    
    def get_count(self) -> int:
        """
        Returns the number of set bits (row IDs).
        
        Returns:
            int: Count of set bits
        """
        if self._use_bitarray:
            return self.bits.count(1)  # Count set bits - optimized in bitarray
        else:
            return len(self.row_ids)
    
    @classmethod
    def from_base64(cls, encoded_string: str, universe_size: Optional[int] = None) -> 'BitVector':
        """
        Creates BitVector from Base64 encoded string.
        
        Args:
            encoded_string (str): Base64 encoded data
            universe_size (int, optional): Universe size for bitarray mode
            
        Returns:
            BitVector: New BitVector instance
        """
        try:
            # Decode base64 to get the raw data
            decoded_data = base64.b64decode(encoded_string)
            
            # Check for format headers
            if decoded_data.startswith(b'BITARRAY:'):
                if not BITARRAY_AVAILABLE:
                    raise ValueError("BitArray format requires bitarray package")
                
                # Extract universe size and bit positions
                header_len = len(b'BITARRAY:')
                stored_universe_size = int.from_bytes(decoded_data[header_len:header_len+4], byteorder='big')
                pickled_bits = decoded_data[header_len+4:]
                
                # Use stored universe size if not provided
                if universe_size is None:
                    universe_size = stored_universe_size
                elif universe_size != stored_universe_size:
                    raise ValueError(f"Universe size mismatch: expected {stored_universe_size}, got {universe_size}")
                
                # Unpickle the set bit positions
                set_bits = pickle.loads(pickled_bits)
                if not isinstance(set_bits, list):
                    raise ValueError("Expected list of bit positions")
                
                # Create BitVector and set the specific bits
                result = cls(universe_size=universe_size)
                for bit_pos in set_bits:
                    if 0 <= bit_pos < universe_size:
                        result.bits[bit_pos] = 1
                    else:
                        raise ValueError(f"Bit position {bit_pos} outside universe size {universe_size}")
                
                return result
                
            elif decoded_data.startswith(b'SETDATA:'):
                # Extract and unpickle the set data
                header_len = len(b'SETDATA:')
                pickled_data = decoded_data[header_len:]
                row_ids = pickle.loads(pickled_data)
                
                if not isinstance(row_ids, set):
                    raise ValueError("Decoded data is not a set")
                
                if not all(isinstance(row_id, int) for row_id in row_ids):
                    raise ValueError("All row IDs must be integers")
                    
                return cls(row_ids, universe_size)
            
            else:
                # Try legacy format (no header) - assume it's pickled set
                try:
                    row_ids = pickle.loads(decoded_data)
                    if isinstance(row_ids, set) and all(isinstance(rid, int) for rid in row_ids):
                        return cls(row_ids, universe_size)
                    else:
                        raise ValueError("Legacy format: not a valid set of integers")
                except Exception:
                    raise ValueError("Unknown format: data doesn't start with expected headers and isn't legacy pickle format")
                
        except Exception as e:
            raise ValueError(f"Invalid base64 encoded BitVector: {e}")
    
    def to_base64(self) -> str:
        """
        Serializes BitVector to Base64 encoded string.
        
        Returns:
            str: Base64 encoded representation
        """
        if self._use_bitarray:
            # For bitarray mode, store the set bit positions explicitly
            # This is more reliable than tobytes()/frombytes()
            set_bits = [i for i in range(len(self.bits)) if self.bits[i]]
            
            # Format: header + universe_size + set of bit positions
            header = b'BITARRAY:'
            universe_bytes = self.universe_size.to_bytes(4, byteorder='big')
            pickled_bits = pickle.dumps(set_bits)
            
            full_data = header + universe_bytes + pickled_bits
            encoded_string = base64.b64encode(full_data).decode('utf-8')
            return encoded_string
        else:
            # Serialize set using pickle with header
            header = b'SETDATA:'
            pickled_data = pickle.dumps(self.row_ids)
            full_data = header + pickled_data
            
            encoded_string = base64.b64encode(full_data).decode('utf-8')
            return encoded_string
    
    def add_row_id(self, row_id: int) -> None:
        """
        Adds a row ID to this BitVector.
        
        Args:
            row_id (int): Row ID to add
        """
        if self._use_bitarray:
            if 0 <= row_id < self.universe_size:
                self.bits[row_id] = 1
            else:
                raise ValueError(f"Row ID {row_id} outside universe size {self.universe_size}")
        else:
            self.row_ids.add(row_id)
    
    def __eq__(self, other) -> bool:
        """
        Checks equality with another BitVector.
        """
        if not isinstance(other, BitVector):
            return False
        
        if self._use_bitarray and other._use_bitarray:
            return self.bits == other.bits
        elif not self._use_bitarray and not other._use_bitarray:
            return self.row_ids == other.row_ids
        else:
            return False
    
    def __repr__(self) -> str:
        """
        String representation of BitVector.
        """
        if self._use_bitarray:
            set_bits = [i for i in range(len(self.bits)) if self.bits[i]]
            return f"BitVector(bits={set_bits}, universe_size={self.universe_size})"
        else:
            return f"BitVector(row_ids={sorted(self.row_ids)})"


# Register BitVector with the factory
SketchFactory.register_sketch_type('bitvector', BitVector)
