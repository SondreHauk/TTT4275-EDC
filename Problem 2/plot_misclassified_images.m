function plot_misclassified_images(test_label, test_data, nearest_neighboor, N_pictures_to_plot, misclassified)
    N_te = size(test_data, 1);
    test_idx = 1:N_te;

    if misclassified == true
        classified_idx = test_idx(nearest_neighboor ~= test_label);
    else
        classified_idx = test_idx(nearest_neighboor == test_label);
    end

    for i=1:N_pictures_to_plot
        misclassified_image = zeros(28,28);
        misclassified_image(:) = test_data(classified_idx(i), :);
        image(misclassified_image');
        clc;
        fprintf("Picture number: %d\n", i);
        fprintf("Classifier says it is a: %d\n", nearest_neighboor(classified_idx(i)));
        fprintf("The real value is a: %d\n", test_label(classified_idx(i)));
    end
end