#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <algorithm>

using namespace std;

/**
 *  Проверяет сходимость метода итераций по норме максимального отклонения.
 * Arg: vk Текущий вектор приближений.
 * Arg: vkp Предыдущий вектор приближений.
 * Arg: eps Заданная точность.
 * Returns: true, если разность векторов меньше заданной точности, иначе false.
 */
bool converge(const vector<double>& vk, const vector<double>& vkp, double eps)
{
    double maxDiff = 0;
    for (size_t i = 0; i < vk.size(); ++i)
    {
        maxDiff = max(maxDiff, fabs(vk[i] - vkp[i]));
    }
    return maxDiff < eps;
}

/**
 *  Вычисляет невязку системы уравнений.
 * Arg: a Матрица коэффициентов системы.
 * Arg: v Вектор решений.
 * Arg: b Вектор свободных членов.
 * Returns: Вектор невязки.
 */
vector<double> compute_residual(const vector<vector<double>>& a, const vector<double>& v, const vector<double>& b)
{
    vector<double> residual(b.size(), 0.0);
    for (size_t i = 0; i < a.size(); ++i)
    {
        for (size_t j = 0; j < a[i].size(); ++j)
        {
            residual[i] += a[i][j] * v[j];
        }
        residual[i] -= b[i];
    }
    return residual;
}

/**
 *  Вычисляет координату y по индексу сеточной точки.
 * Arg: j Индекс точки по оси y.
 * Arg: m Количество внутренних узлов по оси y.
 * Arg: c_bound Левая граница по оси y.
 * Arg: d_bound Правая граница по оси y.
 * Returns: Координата y.
 */
double y(int j, int m, double c_bound, double d_bound)
{
    double k = (d_bound - c_bound) / (m + 1);
    return c_bound + j * k;
}

/**
 * Вычисляет координату x по индексу сеточной точки.
 * Arg: i Индекс точки по оси x.
 * Arg: n Количество внутренних узлов по оси x.
 * Arg: a_bound Левая граница по оси x.
 * Arg: b_bound Правая граница по оси x.
 * Returns: Координата x.
 */
double x(int i, int n, double a_bound, double b_bound)
{
    double h = (b_bound - a_bound) / (n + 1);
    return a_bound + i * h;
}

/**
 *  Истинное решение функции u(x, y).
 * Arg: x_val Координата x.
 * Arg: y_val Координата y.
 * Returns: Значение функции u в точке (x, y).
 */
double u(double x_val, double y_val)
{
    return pow(x_val, 3) + pow(y_val, 2) + 3;
}

/**
 *  Функция для вычисления f(x, y) = Δu(x, y).
 * Arg: x_val Координата x.
 * Arg: y_val Координата y.
 * Returns: Значение функции f в точке (x, y).
 */
double f(double x_val, double y_val)
{
    // Δu = d²u/dx² + d²u/dy²
    // Для u = x^3 + y^2 + 3, получаем Δu = 6x + 2
    return -6 * x_val - 2;
}

/**
 * Вычисляет ошибку решения.
 * Arg: v Вектор решения.
 * Arg: n_internal Количество внутренних узлов по оси x.
 * Arg: m_internal Количество внутренних узлов по оси y.
 * Arg: a_bound Левая граница области по x.
 * Arg: b_bound Правая граница области по x.
 * Arg: c_bound Нижняя граница области по y.
 * Arg: d_bound Верхняя граница области по y.
 * Returns: Вектор ошибки.
 */
vector<double> compute_error(const vector<double>& v, int n_internal, int m_internal, double a_bound, double b_bound, double c_bound, double d_bound)
{
    vector<double> error(v.size(), 0.0);
    int h = (b_bound - a_bound) / (n_internal + 1);
    int k = (d_bound - c_bound) / (m_internal + 1);
    for (size_t i = 0; i < v.size(); ++i)
    {   
        int row = i / n_internal + 1;
        int col = i % n_internal + 1;
        error[i] = fabs(v[i] - u(x(col, n_internal, a_bound, b_bound), y(row, m_internal, c_bound, d_bound)));
    }
    return error;
}


/**
 *  Выводит матрицу A в виде блоков 4x4 для лучшей читаемости.
 * Arg: a Матрица коэффициентов системы.
 * Arg: blockSize Размер блока (например, 4).
 */
void print_matrix(const vector<vector<double>>& a, int blockSize)
{
    int n = a.size();
    cout << "\nМатрица A:\n\n";

    for (int i = 0; i < n; ++i)
    {
        // Разделение на блоки по строкам
        if (i % blockSize == 0 && i != 0)
        {
            cout << "\n";
        }

        for (int j = 0; j < a[i].size(); ++j)
        {
            // Разделение на блоки по столбцам
            if (j % blockSize == 0 && j != 0)
            {
                cout << " | ";
            }

            cout << setw(10) << fixed << setprecision(4) << a[i][j] << " ";
        }
        cout << "\n";
    }

    cout << "\n";
}

/**
 *  Выводит вектор невязки и её нормы.
 * Arg: residual Вектор невязки.
 */
void print_residual(const vector<double>& residual)
{
    cout << "\nНевязка (r):\n";
    for (double r : residual)
    {
        cout << setw(15) << std::uppercase << std::scientific << r << "\n";
    }
    double maxResidual = *max_element(residual.begin(), residual.end(), [](double a, double b) {return fabs(a) < fabs(b);});
    cout << "Максимальная невязка: " << std::uppercase << std::scientific << maxResidual << "\n";

    double evk_norm = 0;
    for (double r : residual)
    {
        evk_norm += r * r;
    }
    evk_norm = sqrt(evk_norm);
    cout << "Евклидова норма невязки: " << std::uppercase << std::scientific << evk_norm << "\n";
}

/**
 * Выводит вектор ошибки и её нормы.
 * Arg: error Вектор ошибки.
 */
void print_error(const vector<double>& error)
{
    cout << "\nВектор погрешности:\n";
    for (double e : error)
    {
        cout << e << "\n";
    }
    double maxError = *max_element(error.begin(), error.end(), [](double a, double b) {return fabs(a) < fabs(b);});
    cout << "Максимальная погрешность: " << std::uppercase << std::scientific << maxError << "\n";
}

void print_separator() {
    cout << "\n_________________________________________________\n\n";
}

/**
 *  Реализация метода Зейделя для решения системы линейных уравнений.
 * Arg: a Матрица коэффициентов системы.
 * Arg: b Вектор свободных членов.
 * Arg: n_internal Количество внутренних узлов по оси x.
 * Arg: m_internal Количество внутренних узлов по оси y.
 * Arg: a_bound Левая граница области по x.
 * Arg: b_bound Правая граница области по x.
 * Arg: c_bound Нижняя граница области по y.
 * Arg: d_bound Верхняя граница области по y.
 * Arg: eps Заданная точность.
 * Arg: maxIterations Максимальное количество итераций.
 * Returns: Вектор решений системы.
 */
vector<double> zeidel(
    const vector<vector<double>>& a,
    const vector<double>& b,
    int n_internal,
    int m_internal,
    double a_bound,
    double b_bound,
    double c_bound,
    double d_bound,
    double eps,
    int maxIterations
)
{
    int n = n_internal * m_internal;
    vector<vector<double>> v(n_internal + 2, vector<double>(m_internal + 2, 0.0));
    vector<vector<double>> f_matrix(n_internal + 2, vector<double>(m_internal + 2, 0.0));

    // Заполнение f_matrix из вектора b и граничных условий
    for (int i = 0; i < n; ++i) {
        int row = i / n_internal + 1 ; // +1, т.к. у нас есть фиктивные границы
        int col = i % n_internal + 1; // +1, т.к. у нас есть фиктивные границы
        f_matrix[col][row] = b[i];
    }
   
    double eps_max = 0;
    double eps_cur = 0;
    double v_old, v_new;
    bool f = false;
    int S = 0;

    double h = (b_bound - a_bound) / (n_internal + 1);
    double k = (d_bound - c_bound) / (m_internal + 1);
    double h2 = 1.0 / (h * h);
    double k2 = 1.0 / (k * k);
    double a2 = -2 * (h2 + k2);


    while (!f) {
        eps_max = 0;
        for (int j = 1; j <= m_internal; ++j) {
            for (int i = 1; i <= n_internal; ++i) {
                v_old = v[i][j];
                v_new = -(h2 * (v[i + 1][j] + v[i - 1][j]) + k2 * (v[i][j + 1] + v[i][j - 1]));
                v_new = (v_new + f_matrix[i][j]) / a2;
                eps_cur = fabs(v_old - v_new);
                if (eps_cur > eps_max) {
                    eps_max = eps_cur;
                }
                v[i][j] = v_new;
            }
        }
        S++;
        if ((eps_max <= eps)) {
            cout << "Метод Зейделя сошёлся за " << S << " итераций, достигнув точности " << std::uppercase << std::scientific << eps_max << "\n";
            f = true;
        }

        if ((S >= maxIterations)) {
            cout << "Метод Зейделя не сошёлся за " << S << " итераций, достигнув точности " << std::uppercase << std::scientific << eps_max << "\n";
            f = true;
        }
    }


    vector<double> result(n);
    for (int i = 0; i < n; ++i)
    {
      int row = i / n_internal + 1;
      int col = i % n_internal + 1;
      result[i] = v[col][row];
    }

    return result;
}

int main()
{
    setlocale(LC_ALL, "RU");

    // Параметры сетки
    int n_internal, m_internal;
    double a_bound, b_bound, c_bound, d_bound, eps;
    int maxIterations;

    cout << "Количество внутренних узлов по оси x: ";
    cin >> n_internal;
    cout << "Количество внутренних узлов по оси y: ";
    cin >> m_internal;
    cout << "Границы по оси x (a_bound, b_bound): ";
    cin >> a_bound >> b_bound;
    cout << "Границы по оси y (c_bound, d_bound): ";
    cin >> c_bound >> d_bound;
    cout << "Точность (eps): ";
    cin >> eps;
    cout << "Максимальное число итераций: ";
    cin >> maxIterations;

    double h = (b_bound - a_bound) / (n_internal + 1); // Шаг по x
    double k_step = (d_bound - c_bound) / (m_internal + 1); // Шаг по y

    cout << "Параметры сетки:\n";
    cout << "Количество внутренних узлов по оси x: " << n_internal << "\n";
    cout << "Количество внутренних узлов по оси y: " << m_internal << "\n";
    cout << "Шаг по x (h): " << h << "\n";
    cout << "Шаг по y (k): " << k_step << "\n\n";

    double one_div_hh = 1 / (h * h); // 1/h^2
    double one_div_kk = 1 / (k_step * k_step); // 1/k^2
    double A = -2 * (one_div_hh + one_div_kk); // Диагональный элемент матрицы A

    // Размерность задачи
    int n = n_internal * m_internal; // Общее количество внутренних узлов

    // Инициализация матрицы A и вектора b
    vector<vector<double>> a(n, vector<double>(n, 0));
    vector<double> b_vec(n, 0.0);

    // Заполнение матрицы A
    for (int i = 0; i < n; ++i)
    {
        a[i][i] = A; // Диагональный элемент

        // Соединения по оси y (верх/низ)
        if (i >= n_internal)
        {
            a[i][i - n_internal] = one_div_kk;
        }
        if (i + n_internal < n)
        {
            a[i][i + n_internal] = one_div_kk;
        }

        // Соединения по оси x (лево/право)
        if (i % n_internal != 0)
        {
            a[i][i - 1] = one_div_hh;
        }
        if ((i + 1) % n_internal != 0)
        {
            a[i][i + 1] = one_div_hh;
        }
    }

    // Заполнение вектора b с учетом граничных условий
    for (int i = 0; i < n; ++i)
    {
        int row = i / n_internal;
        int col = i % n_internal;

        // Вычисляем координаты внутреннего узла
        double x_val = a_bound + (col + 1) * h;
        double y_val = c_bound + (row + 1) * k_step;

        // Правая часть уравнения
        b_vec[i] = -f(x_val, y_val);

        // Учет граничных условий
        // Верхняя граница (y = d_bound)
        if (row == m_internal - 1)
        {
            b_vec[i] -= one_div_kk * u(x_val, d_bound);
        }
        // Нижняя граница (y = c_bound)
        if (row == 0)
        {
            b_vec[i] -= one_div_kk * u(x_val, c_bound);
        }
        // Правая граница (x = b_bound)
        if (col == n_internal - 1)
        {
            b_vec[i] -= one_div_hh * u(b_bound, y_val);
        }
        // Левая граница (x = a_bound)
        if (col == 0)
        {
            b_vec[i] -= one_div_hh * u(a_bound, y_val);
        }
    }

    // Вывод матрицы A в виде блоков 4x4
    print_matrix(a, n_internal);

    // Вывод вектора b
    cout << "Вектор b:\n";
    print_separator();
    for (int i = 0; i < b_vec.size(); ++i)
    {
        cout << "b[" << i << "] = " << fixed << setprecision(4) << setw(10) << b_vec[i] << "\n";
    }
    print_separator();

    // Выполнение метода Зейделя
    vector<double> v = zeidel(a, b_vec, n_internal, m_internal, a_bound, b_bound, c_bound, d_bound, eps, maxIterations);

    // Вывод результата
    cout << "\nРезультат (вектор v):\n";
    print_separator();
    for (int i = 0; i < v.size(); ++i)
    {
        cout << "v[" << i << "] = " << fixed << setprecision(6) << setw(15) << v[i] << "\n";
    }
    print_separator();

    // Вычисление и вывод невязки
    vector<double> residual = compute_residual(a, v, b_vec);
    print_residual(residual);

    // Вычисление и вывод ошибки
    vector<double> error = compute_error(v, n_internal, m_internal, a_bound, b_bound, c_bound, d_bound);
    print_error(error);

    return 0;
}