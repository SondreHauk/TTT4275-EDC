function plot_misclassified_images(test_label, test_data, nearest_neighboor, N_pictures_to_plot)
    N_te = size(test_data, 1);
    test_idx = 1:N_te;
    misclassified_idx = test_idx(nearest_neighboor == test_label);
    
    for i=1:N_pictures_to_plot
        misclassified_image = zeros(28,28);
        misclassified_image(:) = test_data(misclassified_idx(i), :);
        image(misclassified_image');
        clc;
        fprintf("Picture number: %d\n", i);
        fprintf("Classifier says it is a: %d\n", nearest_neighboor(misclassified_idx(i)));
        fprintf("The real value is a: %d\n", test_label(misclassified_idx(i)));
        a=1;
        % pause(3);
    end
end