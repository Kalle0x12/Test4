#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <cstdlib>
#include <paralution.hpp>

namespace py = pybind11;
using namespace std;
using namespace paralution;

template<typename T>
void solution(py::array_t<T, py::array::c_style> values, py::array_t<int> columns,
        py::array_t<int> index, py::array_t<T, py::array::c_style> x_vec, py::array_t<T, py::array::c_style> b_vec,
        int info, double abs_tol, double rel_tol, double div_tol, int max_iter) {

    double tock, tick, start;
    int nnz, len_x;

    init_paralution();
    cout << "accel: " << _paralution_available_accelerator() << endl;
    py::buffer_info info_values = values.request();
    auto val = static_cast<T *> (info_values.ptr);
    nnz = info_values.shape[0];
    py::buffer_info info_columns = columns.request();
    auto col = static_cast<int *> (info_columns.ptr);
    py::buffer_info info_index = index.request();
    auto row_offsets = static_cast<int *> (info_index.ptr);
    py::buffer_info info_x = x_vec.request();
    auto xptr = static_cast<T *> (info_x.ptr);
    len_x = info_x.shape[0];
    py::buffer_info info_b = b_vec.request();
    auto bptr = static_cast<T *> (info_b.ptr);
    start = paralution_time();
    if (info == 1) info_paralution();
    tock = paralution_time();
    cout << "Solver  init: " << (tock - start) / 1000000 << " sec" << endl;
    //initialize paralution vector/matrix objects
    LocalVector<T> x;
    LocalVector<T> b;
    LocalMatrix<T> A;
    //deep copy
    //mat.AllocateCSR("A", nnz, len_x, len_x);
    //mat.CopyFromCSR(ind, col, val);

    //shallow copy
    //call setDataPtr* for all objects passed from input
    //the passed pointers will be set to NULL!!
    //hence it is important to call LeaveDataPtr* for all!! these objects
    //to get back there original values before this function returns
    //otherwise python will segfault

    cout << "col: " << col << endl;
    cout << "val: " << val << endl;
    cout << "row_offsets: " << row_offsets << endl;
    cout << "A" << endl;
    A.SetDataPtrCSR(&row_offsets, &col, &val, "A", nnz, len_x, len_x);
    cout << "col: " << col << endl;
    cout << "val: " << val << endl;
    cout << "row_offsets: " << row_offsets << endl;
    A.info();

    cout << "x" << endl;
    // deep copy
    //x.Allocate("x", len_x);
    //x.CopyFromData(xptr);

    cout << "xptr: " << xptr << endl;
    //x.SetDataPtr(&xptr, "x", len_x);
    cout << "xptr: " << xptr << endl;
    x.info();
    cout << "bptr" << endl;
    // deep copy
    //b.Allocate("b", len_x);
    //b.CopyFromData(bptr);
    cout << "bptr: " << bptr << endl;
    //b.SetDataPtr(&bptr, "b", len_x);
    cout << "bptr: " << bptr << endl;
    // MyMoveToAccelerator avoids free() of the buffers passed from python  
    // in stop_paralution()
    cout << "xptr: " << xptr << endl;
    x.MyMoveToAccelerator(&xptr, len_x);
    cout << "xptr: " << xptr << endl;
    b.MyMoveToAccelerator(&bptr, len_x);
    A.MyMoveToAccelerator();
    x.info();
    b.info();
    A.info();

    tick = paralution_time();
    cout << "Solver allocate: " << (tick - tock) / 1000000 << " sec" << endl;
    tock = paralution_time();
    // Linear Solver
    CG<LocalMatrix<T>, LocalVector<T>, T> ls;
    //ls.MoveToAccelerator();
    // Preconditioner
    Jacobi<LocalMatrix<T>, LocalVector<T>, T > p;
    ls.SetOperator(A);
    // initialize linear solver
    ls.Init(abs_tol, rel_tol, div_tol, max_iter);
    ls.SetPreconditioner(p);
    ls.Build();
    ls.Verbose(1);

    tick = paralution_time();
    cout << "Solver setup + Build: " << (tick - tock) / 1000000 << " sec" << endl;

    tock = paralution_time();
    ls.Solve(b, &x);
    tick = paralution_time();
    cout << "Solver Solve:" << (tick - tock) / 1000000 << " sec" << endl;
    tock = paralution_time();

    A.MoveToHost();
    cout << "xptr: " << xptr << endl;
    x.MyMoveToHost(&xptr, len_x);
    cout << "xptr: " << xptr << endl;
    b.MyMoveToHost(&bptr, len_x);
    //deep copy x back to python
    //x.CopyToData(xptr);
    // call LeaveDataPtr* for all!!! objects/buffers that were created by SetDataPtr 
    // if there is no accelerator segfaults will occur else. 
    A.LeaveDataPtrCSR(&row_offsets, &col, &val);
    cout << "xptr: " << xptr << endl;
    for (int i = 0; i < len_x; i++) {
        cout << xptr[i] << endl;
    }
    tick = paralution_time();
    cout << "Solver copy result:" << (tick - tock) / 1000000 << " sec" << endl;

    cout << "Free memory" << endl;
    tock = paralution_time();
    //ls.Clear();
    stop_paralution();
    tick = paralution_time();
    cout << "Solver free + stop:" << (tick - tock) / 1000000 << " sec" << endl;
    cout << "Solver total:" << (tick - start) / 1000000 << " sec" << endl;
}

PYBIND11_PLUGIN(paralution_wrapper) {
    pybind11::module m("paralution_wrapper", "Test paralution interface");
    //selective functions
    //single precision
    m.def("solution", (void(*)(py::array_t<float, py::array::c_style>, py::array_t<int>,
            py::array_t<int>, py::array_t<float, py::array::c_style>, py::array_t<float, py::array::c_style>,
            int, double, double, double, int)) & solution, "Paralution solver for float arrays");
    //double precision
    m.def("solution", (void(*)(py::array_t<double, py::array::c_style>, py::array_t<int>,
            py::array_t<int>, py::array_t<double, py::array::c_style>, py::array_t<double, py::array::c_style>,
            int, double, double, double, int)) & solution, "Paralution solver for double arrays");

    return m.ptr();
}
