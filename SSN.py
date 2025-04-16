import numpy as np
from numpy import linalg as la


def _SSN(Kl, Km, coefficients, mean, measurements, alpha, M, flog):
    # Positivity constraints on K1_\Omega
    posit = True

    # Initialize convergence tolerance
    tol = 1e-14

    # Set maximum number of iterations
    max_iterations = 1000

    # Set initial value for the step length parameter
    theta = 1e-9

    # Length of the coefficient vector
    length = coefficients.size

    # Identity matrix
    Id = np.identity(length + 1)

    # Construct the complete coefficient matrix
    K = np.concatenate((Kl, Km), axis=1)

    # Setup initial q by concatenating coefficients and mean
    point = np.concatenate((coefficients, mean))

    # Compute the misfit vector
    misfit = K @ point - measurements

    if length == 0:
        # Compute the L2 norm of Km
        H = K.T @ M @ K
        # Compute the initial residual vector
        q = point - K.T @ M @ misfit
        # Clip negative entries of q and append the last element
        point = np.append(np.maximum(q[:length], 0), q[length])
        if posit: point[length] = np.maximum(point[length], 0)

        # Iterate
        for i in range(max_iterations):
            # Compute the misfit vector
            misfit = K @ point - measurements
            # Compute the right-hand side of the Newton step
            right_hand = q - point + K.T @ M @ misfit
            # Check if the residual is below the tolerance
            if la.norm(right_hand) <= tol:
                # Update coefficients and mean
                coefficients = np.maximum(q[:length], 0)
                mean = np.array([q[length]])
                if posit: mean = np.maximum(mean, 0)
                # Compute the adjoint vector
                adj = -K.T @ M @ misfit
                # Compute the function value
                F_val = 0.5 * np.dot(misfit, M @ misfit)
                # Compute the optimal value
                opt_val = 0.5 * np.dot(misfit, M @ misfit) + alpha * la.norm(point[:length], 1)
                flog.write("  SSN terminated with residual {} after {} iterations\n".format(la.norm(right_hand), i))
                break

            # Initialize the step length parameter
            theta = theta / 10
            # Construct the diagonal matrix D
            D = np.diag(np.append(np.where(q[:length] > 0, 1, 0), 1))
            if posit: D = np.diag(np.where(q > 0, 1, 0))
            # Construct the modified Hessian matrix
            Mo = Id - D + H @ D
            # Compute the search direction
            direc = la.solve(Mo + theta * Id, right_hand)
            # Update q
            qnew = q - direc
            # Clip negative entries of q and append the last element
            newpoint = np.append(np.maximum(qnew[:length], 0), qnew[length])
            if posit: newpoint[length] = np.maximum(newpoint[length], 0)
            # Compute the misfit vector for the updated point
            newmisfit = K @ newpoint - measurements
            # Compute the directional derivative
            qdiff = 0.5 * np.dot(newmisfit, M @ newmisfit) - 0.5 * np.dot(misfit, M @ misfit)

            # Backtracking line search
            while qdiff >= 1e-3:
                theta = 2 * theta
                direc = la.solve(H + theta * Id, right_hand)
                qnew = q - direc
                newpoint = np.append(np.maximum(qnew[:length], 0), qnew[length])
                if posit: newpoint[length] = np.maximum(newpoint[length], 0)
                newmisfit = K @ newpoint - measurements
                qdiff = 0.5 * np.dot(newmisfit, M @ newmisfit) - 0.5 * np.dot(misfit, M @ misfit)

            # Update q and point for the next iteration
            q = qnew
            point = newpoint
    else:
        # Compute the Hessian matrix for the combined problem
        H = np.block([[Kl.T @ M @ Kl, Kl.T @ M @ Km], [Km.T @ M @ Kl, Km.T @ M @ Km]])
        # Compute the initial residual vector
        q = point - np.concatenate((Kl.T @ M @ misfit + alpha, Km.T @ M @ misfit))
        # Clip negative entries of q and append the last element
        point = np.append(np.maximum(q[:length], 0), q[length])
        if posit: point[length] = np.maximum(point[length], 0)

        # Iterate
        for i in range(max_iterations):
            # Compute the misfit vector
            misfit = K @ point - measurements
            # Compute the right-hand side of the Newton step
            right_hand = q - point + np.concatenate((Kl.T @ M @ misfit + alpha, Km.T @ M @ misfit))
            # Check if the residual is below the tolerance
            if la.norm(right_hand) <= tol:
                # Update coefficients and mean
                coefficients = np.maximum(q[:length], 0)
                mean = np.array([q[length]])
                if posit: mean = np.maximum(mean, 0)
                # Compute the adjoint vector
                adj = -np.concatenate((Kl.T @ M @ misfit, Km.T @ M @ misfit))
                # Compute the function value
                F_val = 0.5 * np.dot(misfit, M @ misfit)
                # Compute the optimal value
                opt_val = 0.5 * np.dot(misfit, M @ misfit) + alpha * la.norm(point[:length], 1)
                flog.write("  SSN terminated with residual {} after {} iterations\n".format(la.norm(right_hand), i))
                break

            # Construct the diagonal matrix D
            D = np.diag(np.append(np.where(q[:length] > 0, 1, 0), 1))
            if posit: D = np.diag(np.where(q > 0, 1, 0))
            # Construct the modified Hessian matrix
            Mo = Id - D + H @ D
            # Initialize the step length parameter
            theta = theta / 10
            # Compute the search direction
            direc = la.solve(Mo + theta * Id, right_hand)
            # Update q
            qnew = q - direc
            # Clip negative entries of q and append the last element
            newpoint = np.append(np.maximum(qnew[:length], 0), qnew[length])
            if posit: newpoint[length] = np.maximum(newpoint[length], 0)
            # Compute the misfit vector for the updated point
            newmisfit = K @ newpoint - measurements
            # Compute the directional derivative
            qdiff = 0.5 * np.dot(newmisfit, M @ newmisfit) + alpha * la.norm(newpoint[:length], 1) \
                    - 0.5 * np.dot(misfit, M @ misfit) - alpha * la.norm(point[:length], 1)

            # Backtracking line search
            while qdiff >= 1e-3:
                theta = 2 * theta
                direc = la.solve(Mo + theta * Id, right_hand)
                qnew = q - direc
                newpoint = np.append(np.maximum(qnew[:length], 0), qnew[length])
                if posit: newpoint[length] = np.maximum(newpoint[length], 0)
                newmisfit = K @ newpoint - measurements
                qdiff = 0.5 * np.dot(newmisfit, M @ newmisfit) + alpha * la.norm(newpoint[:length], 1) \
                        - 0.5 * np.dot(misfit, M @ misfit) - alpha * la.norm(point[:length], 1)

            # Update q and point for the next iteration
            q = qnew
            point = newpoint

    return coefficients, mean, opt_val, F_val
