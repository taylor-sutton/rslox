// heap is our internal interface for allocating objects that can be tracked and GCed. Currently it does not GC,
// merely allocates and tracks.
//
// The entry point is the `Heap` type and its `new_*` methods. These methods allocate a new Lox object and return a reference
// which can be used to access the object: a `HeapRef`. In the future, the `HeapRef` will kept in an alive, usable state as long as it is
// reachable from a GC root; for now it just lives as long as the Heap does. The underlying objects are all owned by the Heap, so they will
// be invalid if he Heap is dropped. It is not undefined behavior to store and access HeapRef that is no longer alive - but it will panic.
// Because of the shared ownership model, values on the heap must be accessed through special APIs.
//
// ```rust
//    use rslox::vm::heap::{Heap, SharedObject, HeapRef}
//    let mut heap = Heap::new();
//    let string_ref = heap.new_string_with_value("my string");
//    // the `map_as_*` method can access the inner value in a convenient way.
//    string_ref.map_as_string(|s| assert_eq!(s, "my string"));
//    // Alternately, the HeapRef can be converted the long way:
//    let shared_obj: SharedObject = heap_ref.as_obj();
//    match shared_obj.borrow() {
//        Object::String(s) => assert_eq!(s, "my string"),
//    };
// ```
//
// Internally, the Heap keeps an ``Rc<RefCell<_>>`` and hands out `Weak<RefCell<_>>`. That way, the heap ref is only alive as long as the heap
// keeps it alive. The HeapRef uses RefCell's runtime borrow checking, meaning that it can be a runtime panic to use incorrectly.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use crate::vm::{Chunk, Value};

#[derive(Debug)]
pub struct InternedString {
    length: usize,
    // Another option might be Rc<str> or Rc<[u8]> - Rc implements some useful From impls
    // (e.g. impl From<String> for Rc<str> and impl From &[T] for Rc<[T]>)
    // That would let us avoid unsafe. But let's rolll with the unsafe to avoid the overhead
    // of ref counting, plus as a learning experience.
    data: *const u8,
}

impl InternedString {
    pub fn as_str(&self) -> &str {
        // SAFETY
        // The only way to get an InternedString outside this module is as a the object inside
        // SharedObject<Rc<RefCell<Object>>
        // and that can only be gotten if the object is still alive.
        // We promise that if the GC has kept an object alive, it has also kept
        // alive any strings the object uses.
        // So long as the GC upholds that, the from_row_parts is safe.
        // The from_utf8_unchecked is safe because the length and data came from a String
        // and there's it isn't modified it once it's been converted to InternedString.
        unsafe {
            let data: &[u8] = std::slice::from_raw_parts(self.data, self.length);
            std::str::from_utf8_unchecked(data)
        }
    }
}

impl Display for InternedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// This impl, really, is the whole point of InternedString:
/// We can check for equality in O(1) by just comparing our pointers.
impl PartialEq for InternedString {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

#[derive(Debug)]
pub struct Function {
    pub arity: usize,
    pub chunk: Chunk,
    pub name: Option<HeapRef>, // If it's the main script, it has no name
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.name.as_ref() {
            None => write!(f, "<script>"),
            Some(name) => write!(f, "<function {}>", &*name.as_obj().borrow()),
        }
    }
}

pub struct NativeFunction {
    func: Box<dyn Fn(&[Value]) -> Value>,
}

impl NativeFunction {
    pub fn new(func: Box<dyn Fn(&[Value]) -> Value>) -> NativeFunction {
        NativeFunction { func }
    }

    pub fn call(&self, values: &[Value]) -> Value {
        (self.func)(values)
    }
}

impl Debug for NativeFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<native fn>")
    }
}

#[derive(Debug)]
pub enum Object {
    InternedString(InternedString),
    Function(Function),
    NativeFunction(NativeFunction),
}

impl Display for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Object::InternedString(i) => Display::fmt(&i, f),
            Object::Function(i) => Display::fmt(&i, f),
            Object::NativeFunction(_) => write!(f, "<native function>"),
        }
    }
}

// Internal representation of a heap object. The objects are arranged in a linked list using the `next` field, and the objects
// are owned by the nodes (with the head node owned by the Heap itself)
#[derive(Debug)]
struct HeapNode {
    next: Option<Box<HeapNode>>,
    object: Rc<RefCell<Object>>,
}

// We hide the implementation details of the shared ownership + interior mutability behind this newtype.
// However, the `borrow` and `borrow_mut` methods mirror (and delegate to) the RefCell methods of the same name.
#[derive(Debug)]
pub struct SharedObject(Rc<RefCell<Object>>);

impl SharedObject {
    pub fn borrow(&self) -> impl Deref<Target = Object> + '_ {
        self.0.borrow()
    }

    // fn borrow_mut(&self) -> impl DerefMut<Target = Object> + '_ {
    //     self.0.borrow_mut()
    // }
}

#[derive(Debug, Clone)]
pub struct HeapRef {
    value: Weak<RefCell<Object>>,
}

#[derive(Debug)]
/// A type for allocating, tracking, and eventually GCing heap-allocated Lox object
pub struct Heap {
    head: Option<Box<HeapNode>>,
    string_table: HashSet<String>,
}

impl Heap {
    /// A new, empty heap.
    pub fn new() -> Heap {
        Heap {
            head: None,
            string_table: HashSet::new(),
        }
    }

    /// New string with value, taking ownership and interning it.
    pub fn new_string(&mut self, value: String) -> HeapRef {
        // Ideally we have the unstable method HashSet::get_or_insert, but alas, that is unstable
        self.string_table.insert(value.clone());
        let string_ref = self.string_table.get(&value).unwrap();
        let o = Rc::new(RefCell::new(Object::InternedString(InternedString {
            data: string_ref.as_ptr(),
            length: string_ref.len(),
        })));
        let obj = HeapNode {
            object: o.clone(),
            next: self.head.take(),
        };
        self.head = Some(Box::new(obj));
        HeapRef {
            value: std::rc::Rc::downgrade(&o),
        }
    }

    pub fn new_function(&mut self, function: Function) -> HeapRef {
        let o = Rc::new(RefCell::new(Object::Function(function)));
        let obj = HeapNode {
            object: o.clone(),
            next: self.head.take(),
        };
        self.head = Some(Box::new(obj));
        HeapRef {
            value: std::rc::Rc::downgrade(&o),
        }
    }

    pub fn new_native(&mut self, function: NativeFunction) -> HeapRef {
        let o = Rc::new(RefCell::new(Object::NativeFunction(function)));
        let obj = HeapNode {
            object: o.clone(),
            next: self.head.take(),
        };
        self.head = Some(Box::new(obj));
        HeapRef {
            value: std::rc::Rc::downgrade(&o),
        }
    }

    // Print out all the objects on the heap in linked list order for debugging.
    #[allow(dead_code)]
    fn dump(&self) {
        let mut next = &self.head;
        while let Some(node) = next {
            match &*node.object.borrow() {
                Object::InternedString(i) => {
                    println!("{}", i.as_str());
                }
                Object::Function(f) => {
                    println!("{}", f);
                }
                Object::NativeFunction(_) => {
                    println!("<native function>");
                }
            }
            next = &node.next
        }
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}

impl Object {
    // pub fn is_string(&self) -> bool {
    //     true
    // }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Object::InternedString(i) => Some(i.as_str()),
            _ => None,
        }
    }

    pub fn as_function(&self) -> Option<&Function> {
        match self {
            Object::Function(f) => Some(f),
            _ => None,
        }
    }
}

impl HeapRef {
    // Convert a ref to an object, panicking if the ref is no longer alive.
    pub fn as_obj(&self) -> SharedObject {
        SharedObject(self.value.upgrade().unwrap())
    }

    // Apply a function to the inner object if it's a string, returning the result if it's a string
    // And none if it isn't a string
    pub fn map_as_string<F, Ret>(&self, f: F) -> Option<Ret>
    where
        F: FnOnce(&str) -> Ret,
    {
        Some(f(self.as_obj().borrow().as_string()?))
    }

    // Apply a function to the inner object if it's a string, returning the result if it's a string
    // And none if it isn't a string
    pub fn map_as_function<F, Ret>(&self, f: F) -> Option<Ret>
    where
        F: FnOnce(&Function) -> Ret,
    {
        Some(f(self.as_obj().borrow().as_function()?))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_heap() {
        let mut heap = Heap::new();
        let n1 = heap.new_string("Goodbye, world".into());
        let n2 = heap.new_string("Hello, world".into());

        n1.map_as_string(|s| assert_eq!(s, "Goodbye, world"));
        n2.map_as_string(|s| assert_eq!(s, "Hello, world"));

        heap.dump();
    }
}
