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
    int nnz, len_xarr;

    py::buffer_info info_values = values.request();
    auto val = static_cast<T *> (info_values.ptr);
    nnz = info_values.shape[0];
    py::buffer_info info_columns = columns.request();
    auto col = static_cast<int *> (info_columns.ptr);
    py::buffer_info info_index = index.request();
    auto ind = static_cast<int *> (info_index.ptr);
    py::buffer_info info_x = x_vec.request();
    auto x_arr = static_cast<T *> (info_x.ptr);
    len_xarr = info_x.shape[0];
    py::buffer_info info_b = b_vec.request();
    auto b_arr = static_cast<T *> (info_b.ptr);

    start = paralution_time();
    init_paralution();
    if (info == 1) info_paralution();
    tock = paralution_time();
    cout << "Solver  init: " << (tock - start) / 1000000 << " sec" << endl;

    LocalVector<T> x;
    LocalVector<T> rhs;
    LocalMatrix<T> mat;

    mat.AllocateCSR("A", nnz, len_xarr, len_xarr);
    mat.CopyFromCSR(ind, col, val);
    mat.MoveToAccelerator();
    mat.info();

    cout << "x" << endl;
    x.Allocate("x", len_xarr);
    x.CopyFromData(x_arr);
    x.MoveToAccelerator();

    cout << "rhs" << endl;
    rhs.Allocate("rhs", len_xarr);
    rhs.CopyFromData(b_arr);
    rhs.MoveToAccelerator();
    tick = paralution_time();
    cout << "Solver allocate: " << (tick - tock) / 1000000 << " sec" << endl;
    tock = paralution_time();
    // Linear Solver
    CG<LocalMatrix<T>, LocalVector<T>, T> ls;
    //CR<LocalMatrix<float>, LocalVector<float>, float> ls; // faster than CG
    //FixedPoint<LocalMatrix<float>, LocalVector<float>, float> ls;  //diverges
    //ls.SetRelaxation(1.5);
    // Preconditioner
    Jacobi<LocalMatrix<T>, LocalVector<T>, T > p; //OK with CUDA
    //ILU<LocalMatrix<float>, LocalVector<float>, float> p;  //needs a lot of memory with CUDA. rel. slow
    //MultiColoredSGS<LocalMatrix<float>, LocalVector<float>, float> p; //LocalMatrix::ExtractSubMatrix() is performed on the host due to size = 1
    //p.SetRelaxation(1.5);
    //SGS<LocalMatrix<float>, LocalVector<float>, float > p; //slow
    //GS<LocalMatrix<float>, LocalVector<float>, float > p; //extremely slow

    //TNS<LocalMatrix<float>, LocalVector<float>, float > p; //
    // explicit (false) or implicit (true)
    //p.Set(true);  //false: CUDA out of memory, true: medium slow

    // Build only on host
    //SPAI<LocalMatrix<float>, LocalVector<float>, float > p;
    //MultiColoredILU<LocalMatrix<float>, LocalVector<float>, float> p;  //does run
    //MultiColoredGS<LocalMatrix<float>, LocalVector<float>, float> p;  //extremely slow
    //p.SetRelaxation(1.9);

    ls.SetOperator(mat);
    // initialize linear solver
    ls.Init(abs_tol, rel_tol, div_tol, max_iter);
    ls.SetPreconditioner(p);
    ls.Build();
    ls.Verbose(1);
    //        mat.MoveToAccelerator();
    //        rhs.MoveToAccelerator();
    //        x.MoveToAccelerator();
    //        ls.MoveToAccelerator();
    tick = paralution_time();
    cout << "Solver setup + Build: " << (tick - tock) / 1000000 << " sec" << endl;

    tock = paralution_time();
    ls.Solve(rhs, &x);
    tick = paralution_time();
    cout << "Solver Solve:" << (tick - tock) / 1000000 << " sec" << endl;
    tock = paralution_time();
    x.MoveToHost();
    x.CopyToData(x_arr);
    tick = paralution_time();
    cout << "Solver copy result:" << (tick - tock) / 1000000 << " sec" << endl;

    cout << "Free memory" << endl;
    tock = paralution_time();
    ls.Clear();
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

