import pytest
import torch
import torch.nn as nn
from manuscript.detectors._east.sam import SAMSolver


# --- Тесты для инициализации SAMSolver ---


def test_sam_solver_initialization_default():
    """Тест инициализации SAMSolver с параметрами по умолчанию"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    assert optimizer.use_adaptive == False
    assert len(optimizer.param_groups) > 0
    assert optimizer.param_groups[0]["rho"] == 0.05
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_sam_solver_initialization_custom_rho():
    """Тест инициализации SAMSolver с кастомным rho"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, rho=0.1, lr=0.01
    )
    
    assert optimizer.param_groups[0]["rho"] == 0.1


def test_sam_solver_initialization_adaptive():
    """Тест инициализации SAMSolver с adaptive режимом"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(),
        base_optimizer_cls=torch.optim.SGD,
        use_adaptive=True,
        lr=0.01,
    )
    
    assert optimizer.use_adaptive == True


def test_sam_solver_initialization_negative_rho():
    """Тест инициализации SAMSolver с отрицательным rho"""
    model = nn.Linear(10, 5)
    
    with pytest.raises(ValueError, match="rho must be non-negative"):
        SAMSolver(
            model.parameters(), base_optimizer_cls=torch.optim.SGD, rho=-0.1, lr=0.01
        )


def test_sam_solver_initialization_adam():
    """Тест инициализации SAMSolver с Adam оптимизатором"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.001
    )
    
    assert isinstance(optimizer._optimizer, torch.optim.Adam)


def test_sam_solver_initialization_different_optimizers():
    """Тест инициализации SAMSolver с разными оптимизаторами"""
    model = nn.Linear(10, 5)
    
    # SGD
    opt_sgd = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    assert isinstance(opt_sgd._optimizer, torch.optim.SGD)
    
    # Adam
    opt_adam = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.001
    )
    assert isinstance(opt_adam._optimizer, torch.optim.Adam)
    
    # RMSprop
    opt_rmsprop = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.RMSprop, lr=0.01
    )
    assert isinstance(opt_rmsprop._optimizer, torch.optim.RMSprop)


# --- Тесты для метода step ---


def test_sam_solver_step_basic():
    """Тест базового шага оптимизации SAMSolver"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    x = torch.randn(3, 10)
    y = torch.randn(3, 5)
    
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    initial_params = [p.clone() for p in model.parameters()]
    loss = optimizer.step(closure)
    
    # Проверяем что параметры изменились
    params_changed = any(
        not torch.equal(p1, p2)
        for p1, p2 in zip(initial_params, model.parameters())
    )
    assert params_changed
    assert isinstance(loss, torch.Tensor)


def test_sam_solver_step_reduces_loss():
    """Тест что SAMSolver уменьшает loss"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.1
    )
    
    x = torch.randn(10, 10)
    y = torch.randn(10, 5)
    
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    initial_loss = closure().item()
    
    # Несколько шагов оптимизации
    for _ in range(10):
        optimizer.step(closure)
    
    final_loss = closure().item()
    
    # Loss должен уменьшиться после нескольких итераций
    assert final_loss < initial_loss


def test_sam_solver_step_with_adaptive():
    """Тест шага SAMSolver в adaptive режиме"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(),
        base_optimizer_cls=torch.optim.SGD,
        use_adaptive=True,
        lr=0.01,
    )
    
    x = torch.randn(3, 10)
    y = torch.randn(3, 5)
    
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    loss = optimizer.step(closure)
    assert isinstance(loss, torch.Tensor)


def test_sam_solver_step_multiple_iterations():
    """Тест множественных итераций SAMSolver"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    x = torch.randn(5, 10)
    y = torch.randn(5, 5)
    
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    losses = []
    for _ in range(5):
        loss = optimizer.step(closure)
        losses.append(loss.item())
    
    # Все loss должны быть валидными
    assert all(l >= 0 for l in losses)


# --- Тесты для zero_grad ---


def test_sam_solver_zero_grad():
    """Тест метода zero_grad"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    x = torch.randn(3, 10)
    y = torch.randn(3, 5)
    
    # Создаём градиенты
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()
    
    # Проверяем что градиенты есть
    assert any(p.grad is not None for p in model.parameters())
    
    # Обнуляем
    optimizer.zero_grad()
    
    # Проверяем что градиенты обнулились
    assert all(p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)) 
               for p in model.parameters())


# --- Тесты для state_dict и load_state_dict ---


def test_sam_solver_state_dict():
    """Тест получения state_dict"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    state = optimizer.state_dict()
    
    assert isinstance(state, dict)
    assert "state" in state or "param_groups" in state


def test_sam_solver_load_state_dict():
    """Тест загрузки state_dict"""
    model = nn.Linear(10, 5)
    optimizer1 = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    # Делаем шаг оптимизации
    x = torch.randn(3, 10)
    y = torch.randn(3, 5)
    
    def closure():
        optimizer1.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    optimizer1.step(closure)
    
    # Сохраняем состояние
    state = optimizer1.state_dict()
    
    # Создаём новый оптимизатор и загружаем состояние
    model2 = nn.Linear(10, 5)
    optimizer2 = SAMSolver(
        model2.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    optimizer2.load_state_dict(state)
    
    # Состояния должны быть одинаковыми
    assert optimizer2.state_dict().keys() == optimizer1.state_dict().keys()


def test_sam_solver_state_persistence():
    """Тест сохранения и восстановления состояния оптимизатора"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.001
    )
    
    x = torch.randn(5, 10)
    y = torch.randn(5, 5)
    
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    # Делаем несколько шагов
    for _ in range(3):
        optimizer.step(closure)
    
    # Сохраняем состояние
    saved_state = optimizer.state_dict()
    
    # Делаем ещё шаги
    for _ in range(3):
        optimizer.step(closure)
    
    # Восстанавливаем состояние
    optimizer.load_state_dict(saved_state)
    
    # Состояние должно быть восстановлено (проверяем структуру)
    current_state = optimizer.state_dict()
    assert current_state.keys() == saved_state.keys()
    assert len(current_state['param_groups']) == len(saved_state['param_groups'])


# --- Тесты для _compute_grad_magnitude ---


def test_sam_solver_compute_grad_magnitude_with_gradients():
    """Тест вычисления grad magnitude с градиентами"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    x = torch.randn(3, 10)
    y = torch.randn(3, 5)
    
    # Создаём градиенты
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()
    
    # Вычисляем grad magnitude
    grad_norm = optimizer._compute_grad_magnitude()
    
    assert isinstance(grad_norm, torch.Tensor)
    assert grad_norm > 0


def test_sam_solver_compute_grad_magnitude_no_gradients():
    """Тест вычисления grad magnitude без градиентов"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    # Нет градиентов
    grad_norm = optimizer._compute_grad_magnitude()
    
    assert isinstance(grad_norm, torch.Tensor)
    assert grad_norm == 0.0


def test_sam_solver_compute_grad_magnitude_adaptive():
    """Тест вычисления grad magnitude в adaptive режиме"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(),
        base_optimizer_cls=torch.optim.SGD,
        use_adaptive=True,
        lr=0.01,
    )
    
    x = torch.randn(3, 10)
    y = torch.randn(3, 5)
    
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()
    
    grad_norm = optimizer._compute_grad_magnitude()
    
    assert isinstance(grad_norm, torch.Tensor)
    assert grad_norm > 0


# --- Интеграционные тесты ---


def test_sam_solver_training_loop():
    """Тест полного цикла обучения с SAMSolver"""
    # Простая задача регрессии
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    # Генерируем данные
    X = torch.randn(100, 5)
    y = torch.randn(100, 1)
    
    def closure():
        optimizer.zero_grad()
        output = model(X)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    initial_loss = closure().item()
    
    # Обучаем
    for _ in range(20):
        loss = optimizer.step(closure)
    
    final_loss = loss.item()
    
    # Loss должен уменьшиться
    assert final_loss < initial_loss


def test_sam_solver_different_model_architectures():
    """Тест SAMSolver с разными архитектурами моделей"""
    x = torch.randn(5, 10)
    y = torch.randn(5, 3)
    
    # Простая линейная модель
    model1 = nn.Linear(10, 3)
    opt1 = SAMSolver(model1.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01)
    
    def closure1():
        opt1.zero_grad()
        loss = nn.functional.mse_loss(model1(x), y)
        return loss
    
    loss1 = opt1.step(closure1)
    assert torch.isfinite(loss1)
    
    # Многослойная модель
    model2 = nn.Sequential(
        nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 3)
    )
    opt2 = SAMSolver(model2.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.001)
    
    def closure2():
        opt2.zero_grad()
        loss = nn.functional.mse_loss(model2(x), y)
        return loss
    
    loss2 = opt2.step(closure2)
    assert torch.isfinite(loss2)


def test_sam_solver_batch_training():
    """Тест SAMSolver с batch обучением"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    # Разные размеры batch
    for batch_size in [1, 5, 10, 20]:
        x = torch.randn(batch_size, 10)
        y = torch.randn(batch_size, 5)
        
        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            return loss
        
        loss = optimizer.step(closure)
        assert torch.isfinite(loss)


def test_sam_solver_rho_values():
    """Тест SAMSolver с разными значениями rho"""
    model = nn.Linear(10, 5)
    x = torch.randn(5, 10)
    y = torch.randn(5, 5)
    
    for rho in [0.01, 0.05, 0.1, 0.5]:
        optimizer = SAMSolver(
            model.parameters(), base_optimizer_cls=torch.optim.SGD, rho=rho, lr=0.01
        )
        
        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            return loss
        
        loss = optimizer.step(closure)
        assert torch.isfinite(loss)


def test_sam_solver_cuda_device():
    """Тест SAMSolver на CUDA устройстве"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = nn.Linear(10, 5).cuda()
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    x = torch.randn(3, 10).cuda()
    y = torch.randn(3, 5).cuda()
    
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    loss = optimizer.step(closure)
    
    assert loss.device.type == "cuda"


def test_sam_solver_cpu_device():
    """Тест SAMSolver на CPU устройстве"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    x = torch.randn(3, 10)
    y = torch.randn(3, 5)
    
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    loss = optimizer.step(closure)
    
    assert loss.device.type == "cpu"


def test_sam_solver_parameter_updates():
    """Тест что параметры обновляются корректно"""
    model = nn.Linear(5, 3)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.1
    )
    
    x = torch.randn(10, 5)
    y = torch.randn(10, 3)
    
    # Сохраняем начальные параметры
    initial_weight = model.weight.clone()
    initial_bias = model.bias.clone()
    
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    # Делаем шаг
    optimizer.step(closure)
    
    # Параметры должны измениться
    assert not torch.equal(model.weight, initial_weight)
    assert not torch.equal(model.bias, initial_bias)


def test_sam_solver_convergence():
    """Тест сходимости SAMSolver на простой задаче"""
    torch.manual_seed(123)
    
    # Создаём простую линейную зависимость
    true_w = torch.randn(5, 1)
    true_b = torch.randn(1)
    
    X = torch.randn(100, 5)
    y = X @ true_w + true_b + torch.randn(100, 1) * 0.1
    
    model = nn.Linear(5, 1)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    
    def closure():
        optimizer.zero_grad()
        output = model(X)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    losses = []
    for _ in range(50):
        loss = optimizer.step(closure)
        losses.append(loss.item())
    
    # Проверяем что loss в целом уменьшается
    # Берём среднее первых 10 и последних 10
    early_avg = sum(losses[:10]) / 10
    late_avg = sum(losses[-10:]) / 10
    
    assert late_avg < early_avg


def test_sam_solver_with_momentum():
    """Тест SAMSolver с SGD + momentum"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(),
        base_optimizer_cls=torch.optim.SGD,
        lr=0.01,
        momentum=0.9,
    )
    
    x = torch.randn(5, 10)
    y = torch.randn(5, 5)
    
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    loss = optimizer.step(closure)
    assert torch.isfinite(loss)


def test_sam_solver_small_model():
    """Тест SAMSolver с маленькой моделью"""
    model = nn.Linear(2, 1)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01  # Уменьшили lr
    )
    
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[3.0], [7.0]])
    
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss
    
    initial_loss = closure().item()
    
    for _ in range(10):
        loss = optimizer.step(closure)
    
    final_loss = loss.item()
    
    # Проверяем что loss конечен и не взорвался
    assert torch.isfinite(torch.tensor(final_loss))
