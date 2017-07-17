#define BUFFER_WIDTH 64
#define BUFFER_HEIGHT 64
#define ZOOM 8
#define WINDOW_WIDTH BUFFER_WIDTH * ZOOM
#define WINDOW_HEIGHT BUFFER_HEIGHT * ZOOM
#define NEURON_LAYER_SIZE BUFFER_WIDTH * BUFFER_HEIGHT
#define LEARNING_CONSTANT 0.5
#define NUMBER_OF_COMMAND 2

#include <iostream>
#include <queue>
#include <cstdlib>
#include <cmath>
#include <Eigen/Dense>
#include <SFML/Graphics.hpp>

float random_float(float a, float b)
{
	float random = (float)rand() / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

void reset_game(sf::RectangleShape& ball, sf::Vector2f& ball_speed, sf::RectangleShape& paddle1, sf::RectangleShape& paddle2)
{
	//Reset ball
	float random = random_float(0.25, 0.75);
	ball_speed.y = random;
	ball_speed.x = 1 - ball_speed.y;
	/*if (rand() % 2)
	{
		ball_speed.x = -ball_speed.x;
	}*/
	if (rand() % 2)
	{
		ball_speed.y = -ball_speed.y;
	}
	ball.setPosition(BUFFER_WIDTH / 2, BUFFER_HEIGHT / 2);

	//Reset paddles
	paddle1.setPosition(0, (BUFFER_HEIGHT / 2) - (paddle1.getSize().y / 2));
	paddle2.setPosition(BUFFER_WIDTH - 1, (BUFFER_HEIGHT / 2) - (paddle1.getSize().y / 2));
}

void get_input_neurons(Eigen::MatrixXf& input_neurons, sf::RectangleShape& ball, sf::RectangleShape& paddle1, sf::RectangleShape& paddle2)
{
	for (size_t i = 0; i < BUFFER_WIDTH; i++)
	{
		for (size_t j = 0; j < BUFFER_HEIGHT; j++)
		{
			input_neurons(i + (j * BUFFER_WIDTH)) = 0;
		}
	}

	size_t x = ball.getPosition().x;
	size_t y = ball.getPosition().y;
	input_neurons(x + (y * BUFFER_WIDTH)) = 1;

	for (size_t i = 0; i < paddle1.getSize().y; i++)
	{
		x = paddle1.getPosition().x;
		y = paddle1.getPosition().y;
		input_neurons(x + ((y + i) * BUFFER_WIDTH)) = 1;
	}
	for (size_t i = 0; i < paddle2.getSize().y; i++)
	{
		x = paddle2.getPosition().x;
		y = paddle2.getPosition().y;
		input_neurons(x + ((y + i) * BUFFER_WIDTH)) = 1;
	}
}

void display_neuron_layer(Eigen::MatrixXf& neuron_layer, sf::Image& neural_image, sf::Texture& neural_texture, sf::Sprite& neural_sprite, sf::RenderWindow& neural_window)
{
	for (size_t j = 0; j < BUFFER_HEIGHT; j++)
	{
		for (size_t i = 0; i < BUFFER_WIDTH; i++)
		{
			float value = neuron_layer(i + (j * BUFFER_WIDTH));
			sf::Color color = sf::Color::White;
			color.a = value * 255;
			neural_image.setPixel(i, j, color);
		}
	}

	neural_texture.update(neural_image);

	neural_window.clear();
	neural_window.draw(neural_sprite);
	neural_window.display();
}

float activation(float input)
{
	return 1 / (1 + exp(-input));
}

void train(std::queue<Eigen::MatrixXf>& inputs_neurons, float value)
{
	Eigen::MatrixXf input_neurons;
	while (inputs_neurons.size() > 0)
	{
		input_neurons = inputs_neurons.front();
		inputs_neurons.pop();
	}
}

int main()
{
	sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Pong");
	sf::RenderWindow neuron_window(sf::VideoMode(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2), "Neural Network View");
	neuron_window.setPosition(sf::Vector2i(0, 0));
	window.setPosition(sf::Vector2i(neuron_window.getPosition().x + neuron_window.getSize().x, neuron_window.getPosition().y));
	window.setFramerateLimit(60); //For human
	sf::View view(sf::FloatRect(0, 0 , BUFFER_WIDTH, BUFFER_HEIGHT));
	window.setView(view);
	neuron_window.setView(view);

	sf::RectangleShape ball(sf::Vector2f(1, 1));
	sf::Vector2f ball_speed;
	sf::RectangleShape paddle1(sf::Vector2f(1, 8));
	sf::RectangleShape paddle2(sf::Vector2f(1, 8));

	sf::Image neural_image;
	sf::Texture neural_texture;
	sf::Sprite neural_sprite;

	neural_image.create(BUFFER_WIDTH, BUFFER_WIDTH, sf::Color::Transparent);
	neural_texture.loadFromImage(neural_image);
	neural_sprite.setTexture(neural_texture);

	reset_game(ball, ball_speed, paddle1, paddle2);

	//TODO: non static neural network structure (library maybe)
	Eigen::MatrixXf input_neurons(1, NEURON_LAYER_SIZE);
	Eigen::MatrixXf output_neurons(1, NEURON_LAYER_SIZE);
	Eigen::MatrixXf command_neurons(1, NUMBER_OF_COMMAND);
	Eigen::MatrixXf hidden_neuron_weights(NEURON_LAYER_SIZE, NEURON_LAYER_SIZE);
	Eigen::MatrixXf hidden_neuron_bias(1, NEURON_LAYER_SIZE);
	Eigen::MatrixXf command_neuron_weights(NEURON_LAYER_SIZE, NUMBER_OF_COMMAND);
	Eigen::MatrixXf command_neuron_bias(1, NUMBER_OF_COMMAND);
	std::queue<Eigen::MatrixXf> inputs_neurons;

	//Init weights
	for (size_t i = 0; i < NEURON_LAYER_SIZE; i++)
	{
		for (size_t j = 0; j < NEURON_LAYER_SIZE; j++)
		{
			hidden_neuron_weights(i, j) = random_float(-1, 1);
		}
	}
	for (size_t i = 0; i < NEURON_LAYER_SIZE; i++)
	{
		for (size_t j = 0; j < NUMBER_OF_COMMAND; j++)
		{
			command_neuron_weights(i, j) = random_float(-1, 1);
		}
	}
	for (size_t i = 0; i < NEURON_LAYER_SIZE; i++)
	{
		hidden_neuron_bias(i) = random_float(-1, 1);
	}
	for (size_t i = 0; i < NUMBER_OF_COMMAND; i++)
	{
		command_neuron_bias(i) = random_float(-1, 1);
	}
	
	int counter = 0;
	sf::Clock clock;
	while (window.isOpen())
	{
		counter++;
		float fps = 1000.0 / clock.restart().asMilliseconds();
		if (counter > 60)
		{
			counter = 0;
			std::cout << "\r" << (int)fps << "          ";
		}

		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				window.close();
			}
		}
		while (neuron_window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				neuron_window.close();
			}
		}

		if (!sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
		{
			/*if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
			{
				if (paddle2.getPosition().y > 0)
				{
					paddle2.move(0, -1);
				}

			}
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
			{
				if (paddle2.getPosition().y + paddle2.getSize().y < BUFFER_HEIGHT)
				{
					paddle2.move(0, 1);
				}
			}
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
			{
				if (paddle1.getPosition().y > 0)
				{
					paddle1.move(0, -1);
				}
			}
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
			{
				if (paddle1.getPosition().y + paddle1.getSize().y < BUFFER_HEIGHT)
				{
					paddle1.move(0, 1);
				}
			}*/
			get_input_neurons(input_neurons, ball, paddle1, paddle2);
			output_neurons = (input_neurons * hidden_neuron_weights) + hidden_neuron_bias;
			output_neurons = output_neurons.unaryExpr(&activation);
			command_neurons = (output_neurons * command_neuron_weights) + command_neuron_bias;
			command_neurons = command_neurons.unaryExpr(&activation);

			//Record game
			inputs_neurons.push(input_neurons);
			//Limit queue
			/*if (inputs_neurons.size() > NEURON_LAYER_SIZE)
			{
				inputs_neurons.pop();
			}*/

			if (command_neurons(0) > 0.5)
			{
				if (paddle2.getPosition().y > 0)
				{
					paddle2.move(0, -1);
				}
			}
			if (command_neurons(1) > 0.5)
			{
				if (paddle2.getPosition().y + paddle2.getSize().y < BUFFER_HEIGHT)
				{
					paddle2.move(0, 1);
				}
			}

			if ((paddle1.getPosition().y > 0) &&
				(ball.getPosition().y < paddle1.getPosition().y + (paddle1.getSize().y / 2)))
			{
				if (rand() % 2)
				{
					paddle1.move(0, -1);
				}
			}
			if ((paddle1.getPosition().y + paddle1.getSize().y < BUFFER_HEIGHT) &&
				(ball.getPosition().y > paddle1.getPosition().y + (paddle1.getSize().y / 2)))
			{
				if (rand() % 2)
				{
					paddle1.move(0, 1);
				}
			}

			ball.move(ball_speed);
			int bpx = ball.getPosition().x;
			int bpy = ball.getPosition().y;
			if ((bpx >= paddle1.getPosition().x) &&
				(bpx + ball.getSize().x <= paddle1.getPosition().x + paddle1.getSize().x) &&
				(bpy >= paddle1.getPosition().y) &&
				(bpy + ball.getSize().y <= paddle1.getPosition().y + paddle1.getSize().y))
			{
				ball.move(-ball_speed);
				ball_speed.y = (ball.getPosition().y - (paddle1.getPosition().y + (paddle1.getSize().y / 2))) / (paddle1.getSize().y * 0.64);
				ball_speed.x = 1 - abs(ball_speed.y);
				ball.move(ball_speed);
				ball.move(ball_speed);
				bpx = ball.getPosition().x;
				bpy = ball.getPosition().y;
			}
			else if ((bpx >= paddle2.getPosition().x) &&
				(bpx + ball.getSize().x <= paddle2.getPosition().x + paddle2.getSize().x) &&
				(bpy >= paddle2.getPosition().y) &&
				(bpy + ball.getSize().y <= paddle2.getPosition().y + paddle2.getSize().y))
			{
				ball.move(-ball_speed);
				ball_speed.y = (ball.getPosition().y - (paddle2.getPosition().y + (paddle2.getSize().y / 2))) / (paddle2.getSize().y * 0.64);
				ball_speed.x = -(1 - abs(ball_speed.y));
				ball.move(ball_speed);
				ball.move(ball_speed);
				bpx = ball.getPosition().x;
				bpy = ball.getPosition().y;
			}
			if (bpy <= 0)
			{
				ball_speed.y = abs(ball_speed.y);
			}
			else if (bpy >= BUFFER_HEIGHT - ball.getSize().y)
			{
				ball_speed.y = -abs(ball_speed.y);
			}
			if (bpx <= 0)
			{
				reset_game(ball, ball_speed, paddle1, paddle2);

				train(inputs_neurons, 1); //Reward AI
			}
			else if (bpx >= BUFFER_WIDTH - ball.getSize().x)
			{
				reset_game(ball, ball_speed, paddle1, paddle2);

				train(inputs_neurons, -1); //Punish AI
			}
		}

		window.clear();
		window.draw(ball);
		window.draw(paddle1);
		window.draw(paddle2);
		window.display();

		display_neuron_layer(input_neurons, neural_image, neural_texture, neural_sprite, neuron_window);
	}

	return EXIT_SUCCESS;
}